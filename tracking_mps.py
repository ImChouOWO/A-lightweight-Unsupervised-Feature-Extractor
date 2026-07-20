# singleproc_video_infer.py
#
# Single-process pipeline:
# - cv2.VideoCapture -> YOLOv7 (+ backbone hook feat) -> ROI Align -> Encoder -> Tracker -> draw -> imshow
#
# Notes:
# - Keeps your original classes/logic as much as possible.
# - Disables multiprocessing/queues to make MPS behavior stable on macOS.
# - TensorRT branch remains available but only used on CUDA.

from __future__ import annotations

import sys
from pathlib import Path
import time
import yaml
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Iterable, Set
from dataclasses import dataclass

import model.utils.tool as tool


# -----------------------------
# Display ID manager (unchanged)
# -----------------------------
@dataclass
class _Entry:
    disp_id: int
    last_seen: int
    born_seq: int


class DisplayIDManager:
    def __init__(self, max_ids: int = 40):
        self.max_ids = int(max_ids)
        self._free_ids = set(range(1, self.max_ids + 1))
        self._map: Dict[int, _Entry] = {}
        self._seq = 0

    def update(self, active_internal_ids: Iterable[int], frame_idx: int) -> None:
        active: Set[int] = set(int(x) for x in active_internal_ids)
        for iid in active:
            ent = self._map.get(iid)
            if ent is not None:
                ent.last_seen = int(frame_idx)
                continue
            disp_id = self._alloc_display_id(frame_idx=int(frame_idx))
            self._seq += 1
            self._map[iid] = _Entry(disp_id=disp_id, last_seen=int(frame_idx), born_seq=self._seq)

    def _alloc_display_id(self, frame_idx: int) -> int:
        if self._free_ids:
            disp_id = min(self._free_ids)
            self._free_ids.remove(disp_id)
            return disp_id
        victim_iid, victim_ent = self._select_victim(frame_idx)
        disp_id = victim_ent.disp_id
        del self._map[victim_iid]
        return disp_id

    def _select_victim(self, frame_idx: int) -> Tuple[int, _Entry]:
        assert len(self._map) > 0
        best_iid: Optional[int] = None
        best_ent: Optional[_Entry] = None
        best_score = None
        for iid, ent in self._map.items():
            staleness = int(frame_idx) - int(ent.last_seen)
            score = (staleness, -ent.born_seq)
            if best_score is None or score > best_score:
                best_score = score
                best_iid = iid
                best_ent = ent
        return best_iid, best_ent  # type: ignore[return-value]

    def get_display_id(self, internal_id: int) -> Optional[int]:
        ent = self._map.get(int(internal_id))
        return None if ent is None else int(ent.disp_id)


# -----------------------------
# Config helpers
# -----------------------------
CONFPATH = "model/conf/conf.yaml"


def load_conf(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _cv2_runtime_tune():
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass


# -----------------------------
# Optional TRT encoder (CUDA only)
# -----------------------------
class TrtEncoderRunner:
    def __init__(self, engine_path: str, input_name: str = "feat", output_name: str = "emb"):
        import tensorrt as trt
        import torch

        self.trt = trt
        self.torch = torch
        self.input_name = input_name
        self.output_name = output_name

        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)

        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        self.stream = torch.cuda.Stream()
        self.out_dtype = self.engine.get_tensor_dtype(self.output_name)
        self.in_shape = tuple(self.engine.get_tensor_shape(self.input_name))

    def __call__(self, x):
        import torch

        if not x.is_cuda:
            raise RuntimeError("TRT input must be a CUDA tensor")
        if not x.is_contiguous():
            x = x.contiguous()

        N, C, H, W = x.shape

        if self.in_shape[0] != -1 and self.in_shape[0] != N:
            raise RuntimeError(f"Engine batch is static: expected {self.in_shape[0]}, got {N}")

        self.context.set_input_shape(self.input_name, (N, C, H, W))

        trt = self.trt
        if self.out_dtype == trt.DataType.FLOAT:
            out = torch.empty((N, 128), device="cuda", dtype=torch.float32)
        elif self.out_dtype == trt.DataType.HALF:
            out = torch.empty((N, 128), device="cuda", dtype=torch.float16)
        else:
            out = torch.empty((N, 128), device="cuda", dtype=torch.float32)

        self.context.set_tensor_address(self.input_name, int(x.data_ptr()))
        self.context.set_tensor_address(self.output_name, int(out.data_ptr()))

        with torch.cuda.stream(self.stream):
            ok = self.context.execute_async_v3(self.stream.cuda_stream)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 failed")

        torch.cuda.current_stream().wait_stream(self.stream)
        return out


# -----------------------------
# Main inference wrapper
# -----------------------------
class MainInfer:
    def __init__(self, yolo_weight: str, ckpt_path: Optional[str] = None, trt_engine_path: Optional[str] = None):
        import torch
        import model.yolov7.yoloDetects2 as yoloDet
        import model.utils.modules.encoderAndHead as encoderAndHead
        from model.mainTracking import Tracking

        self.torch = torch
        # self.device = (
        #     "cuda"
        #     if torch.cuda.is_available()
        #     else "mps"
        #     if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        #     else "cpu"
        # )
        self.device = "cpu"
        self.conf = load_conf(CONFPATH)

        self.use_trt = (trt_engine_path is not None) and (self.device == "cuda")

        if self.use_trt:
            print(f"[Info] Using TensorRT engine: {trt_engine_path}")
            self.encoder = TrtEncoderRunner(str(trt_engine_path), input_name="feat", output_name="emb")
        else:
            print(f"[Info] Using PyTorch encoder model on {self.device}")
            self.encoder = (
                encoderAndHead.Model(
                    in_channels=self.conf["yolo"]["in_channels"],
                    out_channels=self.conf["yolo"]["out_channels"],
                    warmup_epochs=10,
                    proj_dim=128,
                )
                .to(device=self.device, dtype=torch.float32)
                .eval()
            )

            # Only CUDA half
            if self.device == "cuda":
                self.encoder.half()

            if ckpt_path is not None:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                self.encoder.load_state_dict(ckpt["model"], strict=True)

            self.encoder = self.encoder.to(
                device=self.device,
                dtype=torch.float32,
            ).eval()

        # YOLO should be on same device as MainInfer for MPS stability
        self.yolo = yoloDet.YoloDetects(
            weights=yolo_weight,
            conf_thres=self.conf["yolo"]["conf_thres"],
            iou_thres=self.conf["yolo"]["iou_thres"],
            img_size=self.conf["yolo"]["img_size"],
            device=self.device,
        )

        self.tracker = Tracking()

    def roi_align_from_input_boxes(
        self,
        feat,
        boxes_in: List[List[float]],
        input_hw: Tuple[int, int],
        out_size=(10, 10),
        aligned=True,
        sampling_ratio=2,
    ):
        import torch
        from torchvision.ops import roi_align

        H_in, W_in = input_hw
        _, _, Hf, Wf = feat.shape
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
        )


# -----------------------------
# Single-process tracking loop
# -----------------------------
def track_single_process(
    video_path: str,
    out_path: Optional[str] = None,  # kept for signature compatibility
    show_window: bool = True,
    max_ids: int = 40,
    target_size: Optional[Tuple[int, int]] = (1920, 1080),
    min_conf: float = 0.01,
):
    _cv2_runtime_tune()

    conf = load_conf(CONFPATH)
    infer = MainInfer(
        yolo_weight=conf["model"]["yolo_weight"],
        ckpt_path=conf["model"]["encoder_weight"],
        trt_engine_path=conf["model"].get("encoder_trt_engine", None),
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 0

    from tqdm import tqdm
    import torch
    import torch.nn.functional as F

    WIN = "tracking"
    if show_window:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 1280, 720)

    mon = tool.ResourceMonitor(gpu_index=0, sample_interval=0.2).start()
    all_gpu: List[float] = []
    all_cpu: List[float] = []

    id_manager = DisplayIDManager(max_ids=max_ids)
    frame_idx = 0
    cand_gate = int(conf["yolo"].get("nms_candidates", 5))
    BOX_COLOR = (0, 255, 0)

    try:
        with tqdm(total=total_frames if total_frames > 0 else None, desc="Tracking", unit="frame", dynamic_ncols=True) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                if target_size is not None:
                    frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.ndim == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)

                frame = np.ascontiguousarray(frame)

                # --- inference (same as your inference_process, but inline) ---
                bbox_info, _, feat = infer.yolo.run_with_tensor(frame, return_img_tensor=True, cand_gate=cand_gate)

                boxes_in: List[List[float]] = []
                confs: List[float] = []
                boxes_xyxy: List[List[int]] = []

                if (bbox_info is not None) and (len(bbox_info) > 0) and (feat is not None):
                    for d in bbox_info:
                        c = float(d.get("conf", 0.0))
                        if c < float(min_conf):
                            continue

                        b_in = d.get("xyxy_in", None)
                        if b_in is None or len(b_in) != 4:
                            continue

                        boxes_in.append([float(x) for x in b_in])
                        confs.append(c)

                        # draw coords on original frame (fallback from cxcywh)
                        cx, cy, bw, bh = float(d["x"]), float(d["y"]), float(d["w"]), float(d["h"])
                        x1 = int(cx - bw / 2.0)
                        y1 = int(cy - bh / 2.0)
                        x2 = int(cx + bw / 2.0)
                        y2 = int(cy + bh / 2.0)
                        boxes_xyxy.append([x1, y1, x2, y2])

                assignments: List[Dict[str, int]] = []

                if len(boxes_in) == 0 or feat is None:
                    # still update tracker with empty to advance time
                    obj = {"embs": [], "bboxes": [], "confs": [], "input_hw": (frame.shape[0], frame.shape[1]), "frame_id": int(frame_idx)}
                    infer.tracker.update(obj)
                else:
                    input_hw = tuple(bbox_info[0]["input_hw"])

                    # ROI Align (may fail on some MPS torchvision builds; this is the point we want to debug in single-proc)
                    roi = infer.roi_align_from_input_boxes(
                        feat=feat,
                        boxes_in=boxes_in,
                        input_hw=input_hw,
                        out_size=(10, 10),
                    )

                    with torch.no_grad():
                        emb_cur = infer.encoder(roi)  # TRT or torch

                    emb_cur = F.normalize(emb_cur.float(), dim=-1)

                    embs_np = emb_cur.detach().cpu().numpy().astype(np.float32)
                    embs_list = [embs_np[i].reshape(-1) for i in range(embs_np.shape[0])]

                    obj = {
                        "embs": embs_list,
                        "bboxes": boxes_in,
                        "confs": confs,
                        "input_hw": tuple(input_hw),
                        "frame_id": int(frame_idx),
                    }

                    matches_tid, _, _ = infer.tracker.update(obj)
                    assignments = [{"track_id": int(tid), "det_idx": int(det_j)} for (tid, det_j) in (matches_tid or [])]

                # --- draw ---
                for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
                    c = float(confs[i]) if i < len(confs) else 0.0
                    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
                    cv2.putText(
                        frame,
                        f"D{i} conf:{c:.2f}",
                        (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 0),
                        2,
                    )

                active_ids = [a["track_id"] for a in assignments]
                id_manager.update(active_ids, frame_idx)

                for a in assignments:
                    internal_id = int(a["track_id"])
                    det_idx = int(a["det_idx"])
                    display_id = id_manager.get_display_id(internal_id)
                    if display_id is None:
                        continue
                    if det_idx < 0 or det_idx >= len(boxes_xyxy):
                        continue

                    x1, y1, x2, y2 = boxes_xyxy[det_idx]
                    c = float(confs[det_idx]) if det_idx < len(confs) else 0.0

                    cv2.putText(
                        frame,
                        f"ID:{display_id} Conf:{c:.2f}",
                        (x1, max(0, y1 - 28)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 0),
                        3,
                    )

                pbar.set_postfix(det=len(boxes_xyxy), cpu=f"{mon.cpu_util:.0f}%", gpu=f"{mon.gpu_util:.0f}%", dev=infer.device)
                all_gpu.append(mon.gpu_util)
                all_cpu.append(mon.cpu_util)

                if show_window:
                    cv2.imshow(WIN, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                pbar.update(1)
                frame_idx += 1

    finally:
        cap.release()
        mon.close()
        if show_window:
            cv2.destroyAllWindows()

        if all_cpu and all_gpu:
            print(f"[Res]: avg cpu {sum(all_cpu)/len(all_cpu):.2f}% , avg gpu {sum(all_gpu)/len(all_gpu):.2f}%")
            print(f"[Res]: max cpu {max(all_cpu):.2f}% , max gpu {max(all_gpu):.2f}%")


if __name__ == "__main__":
    video_path = "video/car.mp4"
    out_path = None
    track_single_process(video_path, out_path, show_window=True, max_ids=40)
