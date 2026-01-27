# valScr/val.py
import json
from pathlib import Path
from typing import List, Tuple, Optional, Sequence, Dict, Any

import torch


def compute_assoc_metrics(P: torch.Tensor, gt_index: List[int], ks: Sequence[int] = (5, 10)) -> Dict[str, float]:
    """
    P: [Q,N] probability or scores (higher is better). CPU tensor.
    gt_index: length Q, each in [0..N-1] or -1
    ks: e.g. (5,10)
    """
    assert P.dim() == 2, f"P must be 2D [Q,N], got {P.shape}"
    Q, N = P.shape
    assert len(gt_index) == Q, f"gt_index len must match Q. got {len(gt_index)} vs {Q}"

    gt = torch.tensor(gt_index, dtype=torch.long)
    valid = gt >= 0
    num_valid = int(valid.sum().item())

    if num_valid == 0:
        # no matched queries in this pair
        out = {
            "num_queries": float(Q),
            "num_valid": 0.0,
            "top1_acc": 0.0,
            "mean_rank": float("nan"),
            "mrr": float("nan"),
        }
        for k in ks:
            out[f"recall@{int(k)}"] = 0.0
        return out

    P_valid = P[valid]             # [Qv, N]
    gt_valid = gt[valid]           # [Qv]

    # rank: 1..N (1 is best)
    # Use descending sort
    order = torch.argsort(P_valid, dim=1, descending=True)  # [Qv, N]
    # position of GT in the sorted list
    # ranks = where(order == gt)
    gt_expand = gt_valid.view(-1, 1).expand_as(order)
    match = (order == gt_expand)
    # each row has exactly one True if gt in [0..N-1]
    ranks0 = torch.argmax(match.to(torch.int64), dim=1)  # 0-based
    ranks = ranks0 + 1

    # Top-1
    top1_pred = order[:, 0]
    top1_acc = (top1_pred == gt_valid).float().mean().item()

    # Recall@K
    out = {
        "num_queries": float(Q),
        "num_valid": float(num_valid),
        "top1_acc": float(top1_acc),
        "mean_rank": float(ranks.float().mean().item()),
        "mrr": float((1.0 / ranks.float()).mean().item()),
    }
    for k in ks:
        kk = min(int(k), N)
        hit = (ranks <= kk).float().mean().item()
        out[f"recall@{int(k)}"] = float(hit)

    return out


def _load_gt_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
def _orig_xyxy_to_input_xyxy_letterbox(
    xyxy_orig: List[float],
    ratio,                 # from letterbox(): can be float or (rw, rh)
    pad,                   # from letterbox(): (pad_w, pad_h)
) -> List[float]:
    """
    Convert bbox from ORIGINAL image coords -> LETTERBOXED input coords (xyxy_in).
    This matches how YOLOv7 letterbox(frame, auto=False) works.
    """
    x1, y1, x2, y2 = map(float, xyxy_orig)

    if isinstance(ratio, (tuple, list)):
        rw, rh = float(ratio[0]), float(ratio[1])
    else:
        rw = rh = float(ratio)

    padw, padh = float(pad[0]), float(pad[1])

    x1 = x1 * rw + padw
    x2 = x2 * rw + padw
    y1 = y1 * rh + padh
    y2 = y2 * rh + padh

    # ensure x1<=x2, y1<=y2
    xx1, xx2 = min(x1, x2), max(x1, x2)
    yy1, yy2 = min(y1, y2), max(y1, y2)
    return [xx1, yy1, xx2, yy2]


def _convert_gt_objects_to_input_space_letterbox(
    gt_objects: list,
    ratio,
    pad,
) -> list:
    out = []
    for o in gt_objects:
        if not (isinstance(o, dict) and "id" in o and "bbox_xyxy" in o):
            continue
        bbox_in = _orig_xyxy_to_input_xyxy_letterbox(o["bbox_xyxy"], ratio=ratio, pad=pad)
        out.append({"id": int(o["id"]), "bbox_xyxy": bbox_in})
    return out
  
def _scale_xyxy(
    xyxy: List[float],
    src_hw: Tuple[int, int],   # (H0, W0)
    dst_hw: Tuple[int, int],   # (H_in, W_in)
) -> List[float]:
    x1, y1, x2, y2 = map(float, xyxy)
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw

    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)

    x1 *= sx; x2 *= sx
    y1 *= sy; y2 *= sy

    # clamp to dst image bounds
    x1 = max(0.0, min(x1, dst_w - 1.0))
    x2 = max(0.0, min(x2, dst_w - 1.0))
    y1 = max(0.0, min(y1, dst_h - 1.0))
    y2 = max(0.0, min(y2, dst_h - 1.0))

    # ensure x1<=x2, y1<=y2
    xx1, xx2 = min(x1, x2), max(x1, x2)
    yy1, yy2 = min(y1, y2), max(y1, y2)
    return [xx1, yy1, xx2, yy2]


def _convert_gt_objects_to_input_space(
    gt_objects: list,
    src_hw: Tuple[int, int],   # original frame (H0,W0)
    dst_hw: Tuple[int, int],   # yolo input (H_in,W_in)
) -> list:
    """
    Input: objects with bbox_xyxy in ORIGINAL frame pixel coords
    Output: objects with bbox_xyxy scaled to YOLO input-space
    """
    out = []
    for o in gt_objects:
        if not (isinstance(o, dict) and "id" in o and "bbox_xyxy" in o):
            continue
        bbox_in = _scale_xyxy(o["bbox_xyxy"], src_hw=src_hw, dst_hw=dst_hw)
        out.append({"id": int(o["id"]), "bbox_xyxy": bbox_in})
    return out



def _iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return float(inter / union) if union > 0 else 0.0


def _extract_pred_xyxy_from_boxes(boxes_info: list) -> List[List[float]]:
    out = []
    for b in boxes_info:
        if isinstance(b, dict) and "xyxy_in" in b:
            x1, y1, x2, y2 = b["xyxy_in"]
            out.append([float(x1), float(y1), float(x2), float(y2)])
        else:
            out.append([0.0, 0.0, 0.0, 0.0])
    return out


def _assign_pred_to_gt_ids(
    pred_xyxy: List[List[float]],
    gt_objects: list,
    iou_thr: float = 0.5,
    unmatched_id: int = -1,
) -> List[int]:
    if not gt_objects:
        return [unmatched_id] * len(pred_xyxy)

    gt_xyxy = []
    gt_ids = []
    for o in gt_objects:
        if "id" not in o or "bbox_xyxy" not in o:
            continue
        gt_ids.append(int(o["id"]))
        x1, y1, x2, y2 = o["bbox_xyxy"]
        gt_xyxy.append([float(x1), float(y1), float(x2), float(y2)])

    if len(gt_xyxy) == 0:
        return [unmatched_id] * len(pred_xyxy)

    out_ids = []
    for p in pred_xyxy:
        best_iou = 0.0
        best_id = unmatched_id
        for g, gid in zip(gt_xyxy, gt_ids):
            iou = _iou_xyxy(p, g)
            if iou > best_iou:
                best_iou = iou
                best_id = gid
        out_ids.append(best_id if best_iou >= iou_thr else unmatched_id)
    return out_ids


def build_gt_index_from_ids(query_ids: List[int], cand_ids: List[int], unmatched_value: int = -1) -> List[int]:
    id_to_j = {}
    for j, cid in enumerate(cand_ids):
        cid = int(cid)
        if cid == unmatched_value:
            continue
        if cid not in id_to_j:
            id_to_j[cid] = j

    gt_index = []
    for qid in query_ids:
        qid = int(qid)
        if qid == unmatched_value:
            gt_index.append(unmatched_value)
        else:
            gt_index.append(id_to_j.get(qid, unmatched_value))
    return gt_index


def build_gt_index_from_res_and_label_paths(
    res: dict,
    label_pre_path: str,
    label_cur_path: str,
    iou_thr: float = 0.5,
    unmatched_value: int = -1,
) -> Tuple[List[int], List[int], List[int]]:
    data = res["data"]
    A_boxes = data["A_boxes"]  # len=QA (each has xyxy_in, input_hw, and SHOULD have ratio/pad)
    B_boxes = data["B_boxes"]  # len=NB

    gtA_raw = _load_gt_json(label_pre_path)
    gtB_raw = _load_gt_json(label_cur_path)

    A_pred_xyxy = _extract_pred_xyxy_from_boxes(A_boxes)  # xyxy_in
    B_pred_xyxy = _extract_pred_xyxy_from_boxes(B_boxes)

    # --- IMPORTANT: get letterbox ratio/pad for each frame ---
    def _get_ratio_pad(boxes: list):
        if not boxes:
            return None, None
        b0 = boxes[0]
        if not isinstance(b0, dict):
            return None, None
        ratio = b0.get("ratio", None)
        pad = b0.get("pad", None)
        return ratio, pad

    ratioA, padA = _get_ratio_pad(A_boxes)
    ratioB, padB = _get_ratio_pad(B_boxes)

    if ratioA is None or padA is None or ratioB is None or padB is None:
        raise KeyError(
            "Missing ratio/pad in A_boxes/B_boxes. "
            "Please add {'ratio': ratio, 'pad': pad} into yoloDetects2.run_with_tensor() result dict."
        )

    # Convert GT (orig) -> input-space (letterbox)
    gtA = {"objects": _convert_gt_objects_to_input_space_letterbox(gtA_raw.get("objects", []), ratio=ratioA, pad=padA)}
    gtB = {"objects": _convert_gt_objects_to_input_space_letterbox(gtB_raw.get("objects", []), ratio=ratioB, pad=padB)}

    query_ids = _assign_pred_to_gt_ids(
        A_pred_xyxy, gtA.get("objects", []),
        iou_thr=iou_thr, unmatched_id=unmatched_value
    )
    cand_ids = _assign_pred_to_gt_ids(
        B_pred_xyxy, gtB.get("objects", []),
        iou_thr=iou_thr, unmatched_id=unmatched_value
    )

    gt_index = build_gt_index_from_ids(query_ids, cand_ids, unmatched_value=unmatched_value)

    QA, NB = data["P"].shape
    assert len(A_boxes) == QA and len(gt_index) == QA
    assert len(B_boxes) == NB

    return gt_index, query_ids, cand_ids




def infer_paths_to_label_paths(
    frame_pre_path: str,
    frame_cur_path: str,
    datasets_root: str = "tracking/utils/valScr/datasets",
) -> Tuple[str, str]:
    """
    Given:
      .../valScr/datasets/pic/pre/ex_1.jpeg
      .../valScr/datasets/pic/cur/ex_1.jpeg
    Return:
      .../valScr/datasets/lable/pre/ex_1.json
      .../valScr/datasets/lable/cur/ex_1.json

    If you pass absolute paths, it still works if the basename matches.
    """
    datasets_root = Path(datasets_root)
    name_pre = Path(frame_pre_path).stem
    name_cur = Path(frame_cur_path).stem

    label_pre = datasets_root / "lable" / "pre" / f"{name_pre}.json"
    label_cur = datasets_root / "lable" / "cur" / f"{name_cur}.json"
    return str(label_pre), str(label_cur)
