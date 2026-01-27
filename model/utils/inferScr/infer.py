import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))
from tqdm import tqdm
from typing import Dict, Any, List, Tuple, Optional
import dataclasses
import json
import cv2
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
import numpy as np
import model.yolov7.yoloDetects2 as yoloDet
import model.utils.modules.encoderAndHead as encoderAndHead
from model.utils.valScr.val import compute_assoc_metrics
import yaml


import time
CONFPATH = "model/conf/conf.yaml"
def load_conf(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def sanitize_qat_state_dict(
    state_dict: Dict[str, torch.Tensor],
    *,
    strip_prefix: str = "model.",
    drop_prefixes: Tuple[str, ...] = ("quant.",),
    drop_substrings: Tuple[str, ...] = ("fake_quant", "activation_post_process", "observer", "zero_point", "scale"),
    keep_non_tensor: bool = False,
) -> Dict[str, Any]:
    """
    Convert a QAT(wrapper) state_dict to a "clean" FP32-like state_dict.

    Typical inputs:
      - ckpt["model"] saved from QATWrapper / prepared_qat model, keys like:
          "model.rmb.xxx", "quant.activation_post_process.xxx", "...fake_quant..."
    Output:
      - keys match encoderAndHead.Model state_dict, e.g.:
          "rmb.xxx", "head.xxx"
      - drops all fake-quant / observer related entries

    Notes:
      - If your wrapper prefix is not "model.", change strip_prefix accordingly.
      - This function only manipulates keys; tensors remain tensors.
    """
    out: Dict[str, Any] = {}
    keep_exact = {
        "head.logit_scale",
        "head.logit_bias",
        "model.head.logit_scale",
        "model.head.logit_bias",
    }

    for k, v in state_dict.items():
        if k in keep_exact:
            # strip prefix if needed
            kk = k[len(strip_prefix):] if strip_prefix and k.startswith(strip_prefix) else k
            out[kk] = v
            continue
        # optionally skip non-tensor buffers/params (rare in state_dict)
        if (not torch.is_tensor(v)) and (not keep_non_tensor):
            continue

        # drop by prefix
        if any(k.startswith(p) for p in drop_prefixes):
            continue

        # drop by substring (fake-quant / observers / scales etc.)
        if any(s in k for s in drop_substrings):
            continue

        # strip wrapper prefix "model."
        if strip_prefix and k.startswith(strip_prefix):
            k = k[len(strip_prefix):]

        out[k] = v

    return out


def load_sanitized_qat_ckpt_to_fp32_model(
    model: torch.nn.Module,
    ckpt_path: str,
    *,
    map_location: str = "cpu",
    state_key: str = "model",
    strip_prefix: str = "model.",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Load a QAT checkpoint into a FP32 model by sanitizing the state_dict.
    Returns a small report for debugging.
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)
    state = ckpt[state_key] if (isinstance(ckpt, dict) and state_key in ckpt) else ckpt

    cleaned = sanitize_qat_state_dict(
        state,
        strip_prefix=strip_prefix,
    )

    missing, unexpected = model.load_state_dict(cleaned, strict=False)

    report = {
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "num_cleaned_keys": len(cleaned),
        "raw_num_keys": len(state),
    }

    if verbose:
        print(f"[sanitize] raw keys={len(state)} -> cleaned keys={len(cleaned)}")
        if missing:
            print(f"[sanitize] missing_keys (show up to 20): {list(missing)[:20]}")
        if unexpected:
            print(f"[sanitize] unexpected_keys (show up to 20): {list(unexpected)[:20]}")

    return report

class MainInfer:
    def __init__(self, yolo_weight, ckpt_path=None, tau = 0.2):
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
        self.conf = load_conf(path=CONFPATH)
        self.tau = tau
        # encoder 模型
        self.model = encoderAndHead.Model(
            in_channels=self.conf["yolo"]["in_channels"],
            out_channels=self.conf["yolo"]["out_channels"],
            warmup_epochs=10,
            proj_dim=128
        ).to(self.device).eval()
        
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
            try:
                self.model.load_state_dict(state, strict=True)
                print("[load] Loaded as FP32 state_dict (strict=True).")
            except RuntimeError as e:
                print(f"[load] FP32 strict load failed, fallback to sanitize QAT ckpt.\n  err={e}")

                # 移除 model. 前綴 與 fake_quant/observer/quant.* 以便透過 FP32 model 加載 FakeQAT checkpoint
                report = load_sanitized_qat_ckpt_to_fp32_model(
                    model=self.model,
                    ckpt_path=ckpt_path,
                    map_location="cpu",
                    state_key="model",
                    strip_prefix="model.",
                    verbose=True,
                )
                print(f"[load] sanitize report: cleaned={report['num_cleaned_keys']} "
                    f"missing={len(report['missing_keys'])} unexpected={len(report['unexpected_keys'])}")
        if self.device == "cuda":
            self.model.half()
        # YOLOv7 feature extractor
        self.yolo = yoloDet.YoloDetects(
            weights=yolo_weight,
            conf_thres=self.conf["yolo"]["conf_thres"],
            iou_thres=self.conf["yolo"]["iou_thres"],
            img_size=self.conf["yolo"]["img_size"]
        )
        
        

    
    # ---------- helpers ----------
    @staticmethod
    def _load_label(path: str | Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _norm_cxcywh_to_xyxy_orig(
        cx: float, cy: float, w: float, h: float,
        orig_w: int, orig_h: int
    ) -> List[float]:
        # normalized (0..1) in original image -> xyxy in original pixel coords
        x1 = (cx - w / 2) * orig_w
        y1 = (cy - h / 2) * orig_h
        x2 = (cx + w / 2) * orig_w
        y2 = (cy + h / 2) * orig_h
        return [x1, y1, x2, y2]

    @staticmethod
    def _xyxy_orig_to_xyxy_in(
        xyxy_orig: List[float],
        ratio, pad
    ) -> List[float]:
        # letterbox: input = orig * r + pad
        x1, y1, x2, y2 = xyxy_orig
        r = ratio[0] if isinstance(ratio, (tuple, list)) else ratio
        pad_w, pad_h = pad
        return [x1 * r + pad_w, y1 * r + pad_h, x2 * r + pad_w, y2 * r + pad_h]

    @staticmethod
    def _clip_xyxy(xyxy: List[float], W: int, H: int) -> List[float]:
        x1, y1, x2, y2 = map(float, xyxy)
        x1 = max(0.0, min(x1, W - 1.0))
        y1 = max(0.0, min(y1, H - 1.0))
        x2 = max(0.0, min(x2, W - 1.0))
        y2 = max(0.0, min(y2, H - 1.0))
        if x2 <= x1: x2 = min(W - 1.0, x1 + 1.0)
        if y2 <= y1: y2 = min(H - 1.0, y1 + 1.0)
        return [x1, y1, x2, y2]
    @staticmethod
    def _extract_xyxy_in_and_conf(bbox_info: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[float]]:
        """
        From yoloDetects2.run_with_tensor() bbox_info -> (boxes_in, confs)
        boxes_in: list of [x1,y1,x2,y2] in YOLO input coords (letterbox)
        confs:    list of float
        """
        boxes_in: List[List[float]] = []
        confs: List[float] = []
        for d in bbox_info:
            if not isinstance(d, dict):
                continue
            if "xyxy_in" not in d or "conf" not in d:
                continue
            b = d["xyxy_in"]
            if b is None or len(b) != 4:
                continue
            boxes_in.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
            confs.append(float(d["conf"]))
        return boxes_in, confs


    def _extract_feat_and_meta(self, img_bgr: "np.ndarray") -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        _preprocess -> forward 觸發 hook -> 取得 backbone feat
        """
        img_tensor, _, ratio, pad, input_hw = self.yolo._preprocess(img_bgr)

        # 觸發 hook
        self.yolo.backbone_feat = None
        with torch.no_grad():
            _ = self.yolo.model(img_tensor, augment=False)

        feat = self.yolo.backbone_feat
        if feat is None:
            raise RuntimeError("YOLO backbone_feat is None. Check hook layer name/module in yoloDetects2.")

        meta = {
            "orig_hw": tuple(img_bgr.shape[:2]),        # (H0,W0)
            "input_hw": tuple(input_hw),               # (H_in,W_in)
            "ratio": ratio,
            "pad": pad,
            "feat_hw": tuple(feat.shape[-2:]),         # (Hf,Wf)
        }
        return feat, meta

    def _roi_align_from_input_boxes(
        self,
        feat: torch.Tensor,                 # [1,C,Hf,Wf]
        boxes_in: List[List[float]],        # list of [x1,y1,x2,y2] in input coords
        input_hw: Tuple[int, int],          # (H_in,W_in)
        out_size=(7, 7),
        aligned=True,
        sampling_ratio=2,
    ) -> torch.Tensor:
        H_in, W_in = input_hw
        _, _, Hf, Wf = feat.shape

        # 等比例 letterbox 下：用 H 的比例即可(Hf/H_in == Wf/W_in)
        spatial_scale = Hf / float(H_in)

        rois = torch.tensor(
            [[0.0, b[0], b[1], b[2], b[3]] for b in boxes_in],
            dtype=feat.dtype,
            device=feat.device,
        )
        return roi_align(
            input=feat,
            boxes=rois,
            output_size=out_size,
            spatial_scale=spatial_scale,
            sampling_ratio=sampling_ratio,
            aligned=aligned,
        )  # [N,C,7,7]

    # ---------- multi-query inference ----------
    def infer_two_img(
        self,
        frame_cur: str,
        frame_pre: str,
        label_cur_path: str,
        label_pre_path: str,
        tau: float = 0.2,
        topk: int = 10,
        ks=(5, 10),
    ):
        """
        支援多個 query:
          - cur: Q 個 annotations -> Q queries
          - pre: N 個 annotations -> N candidates
        輸出：
          - P: [Q,N] 機率矩陣(row-softmax)
          - gt_index: 長度 Q(每個 query 的正解 candidate index 或 -1)
          - metrics: compute_assoc_metrics(P, gt_index, ks)
        計算方式：
          - ROI feature -> encoder ->128 dim tensor
          - tensor pair -> L2 norm ->cosine similarity matrics
          - similarity matrics -> Probability matrics ->Top-1、Mean Rank、MRR、Recall@K ...
        指標：
          - Top-1 Accuracy:               順位一的物件是不是正確答案
          - Mean Rank:                    正確目標平均排在第幾名
          - MRR(Mean Reciprocal Rank):    模型把正確答案放在前面的程度
          - Recall@K:                     前 K 名召回率
        """

        # ---- read images ----
        img_cur_bgr = cv2.imread(frame_cur)
        img_pre_bgr = cv2.imread(frame_pre)
        if img_cur_bgr is None or img_pre_bgr is None:
            raise FileNotFoundError(f"Failed to read: {frame_pre}, {frame_cur}")

        # ---- load labels ----
        lab_cur = self._load_label(label_cur_path)
        lab_pre = self._load_label(label_pre_path)

        cur_anns = lab_cur.get("annotations", [])
        pre_anns = lab_pre.get("annotations", [])
        if len(cur_anns) == 0 or len(pre_anns) == 0:
            return None, None

        # ---- backbone feat + meta ----
        feat_pre, meta_pre = self._extract_feat_and_meta(img_pre_bgr)
        feat_cur, meta_cur = self._extract_feat_and_meta(img_cur_bgr)

        # ---- build query boxes (cur) ----
        Hc, Wc = img_cur_bgr.shape[:2]
        q_ids: List[int] = []
        q_boxes_in: List[List[float]] = []
        for a in cur_anns:
            qid = int(a.get("id", -1))
            b = a["bbox"]
            xyxy_orig = self._norm_cxcywh_to_xyxy_orig(
                b["cx"], b["cy"], b["w"], b["h"], orig_w=Wc, orig_h=Hc
            )
            xyxy_in = self._xyxy_orig_to_xyxy_in(xyxy_orig, meta_cur["ratio"], meta_cur["pad"])
            xyxy_in = self._clip_xyxy(xyxy_in, W=meta_cur["input_hw"][1], H=meta_cur["input_hw"][0])
            q_ids.append(qid)
            q_boxes_in.append(xyxy_in)

        # ---- build candidate boxes (pre) ----
        Hp, Wp = img_pre_bgr.shape[:2]
        cand_ids: List[int] = []
        cand_boxes_in: List[List[float]] = []
        for a in pre_anns:
            cid = int(a.get("id", -1))
            b = a["bbox"]
            xyxy_orig = self._norm_cxcywh_to_xyxy_orig(
                b["cx"], b["cy"], b["w"], b["h"], orig_w=Wp, orig_h=Hp
            )
            xyxy_in = self._xyxy_orig_to_xyxy_in(xyxy_orig, meta_pre["ratio"], meta_pre["pad"])
            xyxy_in = self._clip_xyxy(xyxy_in, W=meta_pre["input_hw"][1], H=meta_pre["input_hw"][0])
            cand_ids.append(cid)
            cand_boxes_in.append(xyxy_in)

        Q = len(q_ids)
        N = len(cand_ids)
        if Q == 0 or N == 0:
            return None, None

        # ---- ROI Align ----
        rq = self._roi_align_from_input_boxes(feat_cur.to(self.device), q_boxes_in, meta_cur["input_hw"])    # [Q,C,7,7]
        rc = self._roi_align_from_input_boxes(feat_pre.to(self.device), cand_boxes_in, meta_pre["input_hw"]) # [N,C,7,7]

        # ---- encoder embeddings ----
        with torch.no_grad():
            eq = self.model(rq)   # [Q,D]
            ei = self.model(rc)   # [N,D]
        # L2 Normalization for cosine similarity
        eq = F.normalize(eq, dim=-1)
        ei = F.normalize(ei, dim=-1)

        # ---- similarity matrix + row softmax ----
        S = torch.matmul(eq, ei.T)                 # [Q,N]
        P = torch.softmax(S / float(tau), dim=1)   # [Q,N]

        # ---- compute metrics ----
        cand_pos = {cid: i for i, cid in enumerate(cand_ids)}
        gt_index = [cand_pos.get(qid, -1) for qid in q_ids]

        metrics = compute_assoc_metrics(P.detach().cpu(), gt_index, ks=ks)

        res = {
            "meta": {
                "frame_pre": frame_pre,
                "frame_cur": frame_cur,
                "tau": float(tau),
                "Q": Q,
                "N": N,
                "pre": meta_pre,
                "cur": meta_cur,
                "eq":eq,
                "ei":ei,
                "P":P
            },
            "data": {
                "P": P.detach().cpu(),     # [Q,N]
                "S": S.detach().cpu(),     # [Q,N]
                "query_ids": q_ids,
                "cand_ids": cand_ids,
                "gt_index": gt_index,      # len=Q
            }
        }
        return res, metrics
    
    

from collections import defaultdict

def print_mean_metrics(metrics_list):
    if len(metrics_list) == 0:
        print("[WARN] No valid metrics to average")
        return

    acc = defaultdict(list)

    # collect
    for m in metrics_list:
        for k, v in m.items():
            acc[k].append(float(v))

    # print mean
    print("\n========== MEAN METRICS ==========")
    for k in sorted(acc.keys()):
        mean_v = sum(acc[k]) / len(acc[k])
        print(f"{k:12s}: {mean_v:.6f}")



def val(path:str = None):
    conf = load_conf(path=CONFPATH)
    infer = MainInfer(
        yolo_weight=conf["model"]["yolo_weight"],
        ckpt_path=conf["model"]["encoder_weight"],
        tau= conf["model"]["tau"]
    )

    root = Path(path)

    cur_pic_dir = root / "now/pic"
    pre_pic_dir = root / "pre/pic"
    cur_lab_dir = root / "now/lable"
    pre_lab_dir = root / "pre/lable"

    res_list = []
    metrics_list = []

    files = sorted(p.stem for p in cur_pic_dir.glob("*.jpg"))

    print(f"[INFO] Found {len(files)} samples")

    for name in files:
        frame_cur = cur_pic_dir / f"{name}.jpg"
        frame_pre = pre_pic_dir / f"{name}.jpg"
        label_cur = cur_lab_dir / f"{name}.json"
        label_pre = pre_lab_dir / f"{name}.json"

        if not (frame_pre.exists() and label_cur.exists() and label_pre.exists()):
            print(f"[SKIP] Missing pair for {name}")
            continue

        try:
            res, metrics = infer.infer_two_img(
                frame_cur=str(frame_cur),
                frame_pre=str(frame_pre),
                label_cur_path=str(label_cur),
                label_pre_path=str(label_pre),
            )
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            continue

        if res is None or metrics is None:
            print(f"[SKIP] Invalid result for {name}")
            continue

        res_list.append(res)
        metrics_list.append(metrics)

        print(f"[OK] {name}: {metrics}")

    print("\n========== DONE ==========")
    print(f"Valid samples: {len(metrics_list)} / {len(files)}")
    print_mean_metrics(metrics_list)
    
if __name__ == "__main__":
    val()
    


