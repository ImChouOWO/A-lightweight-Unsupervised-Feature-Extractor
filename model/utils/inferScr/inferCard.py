import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import os
ROOT = Path(__file__).resolve().parents[3] 
sys.path.append(str(ROOT))
import yolov7.yoloDetects2 as yoloDet
import model.utils.trainingScr.trainingCard as trainingCardMod
import pickle
import time


def save_infer_result(result: dict, pkl_path: str):
    pkl_path = Path(pkl_path)
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(result, f)

class Infer:
    def __init__(
        self,
        weight_path,
        output_size=(10, 10),
        min_conf=0.3,
    ):
        self.weight_path = weight_path
        self.yoloExtractor = yoloDet.YoloDetects(weights=self.weight_path)

        # reuse trainingCard ROI logic (no saving)
        self._roi_helper = trainingCardMod.PreProcess(
            weight_path=self.weight_path,
            dataset_path="",
            preprocess_res_path=""
        )

        self.output_size = output_size
        self.min_conf = float(min_conf)

    

    @torch.no_grad()
    def encode_rois(self, model, rois, device="cuda"):
        """
        rois:
        - query: [C,H,W] or [1,C,H,W]
        - candidates: [N,C,H,W]
        return: embeddings [B,D]
        """
        if rois.dim() == 3:
            rois = rois.unsqueeze(0)
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
        rois = rois.to(device, non_blocking=True)
        model.eval()
        z = model(rois)  # already normalized in ProjectionHead.forward()
        return z

    @torch.no_grad()
    def query_to_candidates_prob(self, model, query_roi, cand_rois, tau=0.2, device="cuda"):
        """
        return:
        probs: [N] (sum=1)
        logits: [N]
        """
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
        q = self.encode_rois(model, query_roi, device=device)      # [1,D]
        c = self.encode_rois(model, cand_rois, device=device)      # [N,D]

        # cosine since vectors are normalized; still safe to normalize again if you want:
        q = F.normalize(q, dim=1)
        c = F.normalize(c, dim=1)

        logits = (q @ c.t()) / tau        # [1,N]
        probs = torch.softmax(logits, dim=1)  # [1,N], sum=1
        return probs.squeeze(0), logits.squeeze(0)

    @torch.no_grad()
    def build_similarity_matrix(self, model, query_rois, cand_rois, tau=0.2, device="cuda"):
        """
        query_rois: [Q,C,H,W]
        cand_rois : [N,C,H,W]
        return:
        P: [Q,N] each row sums to 1
        logits: [Q,N]
        """
        model.eval()
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
        q = self.encode_rois(model, query_rois, device=device)  # [Q,D]
        c = self.encode_rois(model, cand_rois, device=device)   # [N,D]

        q = F.normalize(q, dim=1)
        c = F.normalize(c, dim=1)

        logits = (q @ c.t()) / tau      # [Q,N]
        P = torch.softmax(logits, dim=1)
        return P, logits
    
    @torch.no_grad()
    def _extract_rois_from_frame(
        self,
        frame_bgr,
        device="cuda",
        min_conf=None,
        sampling_ratio=2,
        aligned=True,
        enforce_min_size=1.0,
    ):
        """
        frame_bgr: OpenCV BGR image
        return:
          roi_feats: [N,C,H,W] on same device as img_feat
          bbox_info_used: list of dict for the kept boxes (aligned with roi_feats index)
        """
        if min_conf is None:
            min_conf = self.min_conf

        bbox_info, _, img_feat = self.yoloExtractor.run_with_tensor(
            frame_bgr, return_img_tensor=True
        )
        if not bbox_info:
            return None, None, None, None, None
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
        img_feat = img_feat.to(device, non_blocking=True)

        xyxy_list = []
        used_boxes = []
        input_hw = None
        ratio = None
        pad = None

        for box in bbox_info:
            if not (isinstance(box, dict) and "xyxy_in" in box):
                continue
            if "conf" in box and float(box["conf"]) < float(min_conf):
                continue

            xyxy_list.append(box["xyxy_in"])
            used_boxes.append(box)

            if input_hw is None and "input_hw" in box:
                input_hw = box["input_hw"]

           
            if ratio is None and "ratio" in box:
                ratio = box["ratio"]
            if pad is None and "pad" in box:
                pad = box["pad"]

        if len(xyxy_list) == 0 or input_hw is None:
            return None, None, None, None, None

       

        # bboxes tensor
        bboxes = torch.tensor(xyxy_list, dtype=torch.float32, device=img_feat.device)

        roi_feats = self._roi_helper._preprocess_roi(
            feat=img_feat,
            bboxes_xyxy=bboxes,
            img_hw=input_hw,
            output_size=self.output_size,
            sampling_ratio=sampling_ratio,
            aligned=aligned,
            enforce_min_size=enforce_min_size,
        )

        # 注入 ratio/pad（如果拿得到）
        if ratio is not None and pad is not None:
            for b in used_boxes:
                b.setdefault("ratio", ratio)
                b.setdefault("pad", pad)

        return roi_feats, used_boxes, input_hw, ratio, pad
    
    @torch.no_grad()
    def infer_two_img(
        self,
        model,
        frameA_bgr,
        frameB_bgr,
        tau=0.2,
        device="cuda",
        topk=5,
        min_conf_A=0.3,
        min_conf_B=0.3,
    ):
        """
        Query = ROIs from frame A
        Candidates = ROIs from frame B

        return dict:
          - P: [QA, NB] prob matrix (CPU tensor)
          - logits: [QA, NB] (CPU tensor)
          - A_boxes / B_boxes: bbox_info lists aligned to indices
          - topk for each query in A: indices and probs
        """
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
        # extract A rois
        roiA, boxesA, input_hw_A, ratio_A, pad_A = self._extract_rois_from_frame(
            frameA_bgr, device=device, min_conf=min_conf_A
        )
        # extract B rois
        roiB, boxesB, input_hw_B, ratio_B, pad_B = self._extract_rois_from_frame(
            frameB_bgr, device=device, min_conf=min_conf_B
        )

        if roiA is None or roiB is None:
            return {
                "ok": False,
                "reason": "No valid ROIs in frameA or frameB after filtering.",
                "P": None,
                "logits": None,
                "A_boxes": boxesA,
                "B_boxes": boxesB,
            }

        
        
        P, logits = self.build_similarity_matrix(
            model=model,
            query_rois=roiA,
            cand_rois=roiB,
            tau=float(tau),
            device=device,
        )  # P: [QA, NB]

        # top-k per query
        NB = P.size(1)
        k = min(int(topk), NB)
        if k > 0:
            topk_prob, topk_idx = torch.topk(P, k=k, dim=1)
        else:
            topk_prob = P.new_empty((P.size(0), 0))
            topk_idx  = torch.empty((P.size(0), 0), dtype=torch.long)

        res = {
            "meta": {
                "time": int(time.time()),
                "input_hw_A": input_hw_A,   # (H_in, W_in)
                "input_hw_B": input_hw_B,
                "tau": float(tau),
                "topk": int(topk),
                "device": str(device),
                "min_conf_A": float(min_conf_A) if min_conf_A is not None else None,
                "min_conf_B": float(min_conf_B) if min_conf_B is not None else None,
                "ratio_A": ratio_A,
                "pad_A": pad_A,
                "ratio_B": ratio_B,
                "pad_B": pad_B,

            },
            "data":{"ok": True,
                    "P": P.detach().cpu(),               # [QA, NB]
                    "logits": logits.detach().cpu(),     # [QA, NB]
                    "A_boxes": boxesA,                   # len=QA
                    "B_boxes": boxesB,                   # len=NB
                    "topk_idx": topk_idx.detach().cpu(),   # [QA, k]
                    "topk_prob": topk_prob.detach().cpu(), # [QA, k]
            }
        }
        save_infer_result(
        res,
        f"training/tracking/utils/inferSrc/inferRes/infer_{int(time.time())}.pkl"
        )
        return res
