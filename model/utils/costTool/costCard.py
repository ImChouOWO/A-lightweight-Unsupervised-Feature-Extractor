"""
    cal_cost()：關聯成本整合器（appearance + bbox + confidence）

    功能：
      - 將三種成本矩陣整合成單一總成本矩陣 C_total，用於：
        * Hungarian baseline assignment（輸入 [M,N] cost matrix）
        * PSO / 其他搜尋器的 fitness 評估（可選輸入 assign 取得 scalar cost）
        * 並容許擴充新的成本矩陣
      - 成本設計為「越小越好」。

    輸入：
      - C_app: Tensor [M, N]
          * 外觀成本矩陣（通常由 P→cost 得到，如 C_app = -log(P + eps)）
          * M=tracks 數、N=detections 數
      - boxes_prev: List[List[float]]，長度 M，每個為 [x1,y1,x2,y2]
      - boxes_cur : List[List[float]]，長度 N，每個為 [x1,y1,x2,y2]
      - input_hw: (H, W)
          * 用於 bbox center distance 的對角線長度正規化（使尺度與解析度無關）
      - conf_prev: List[float]，長度 M
          * track 的歷史 confidence（例如 conf_ema 或上一幀 det conf）
      - conf_cur : List[float]，長度 N
          * 當前 detections 的 confidence

      - w_app / w_bbox / w_conf:
          * 各成本項的線性加權係數
          * 最終：C_total = w_app*C_app + w_bbox*C_bbox + w_conf*C_conf

      - alpha / beta:
          * bbox 成本內部的子權重
          * C_bbox = alpha * center_cost + beta * scale_cost

      - assign（可選）: List[int]，長度 M
          * assign[i] = j 代表 track i 指派到 detection j
          * assign[i] = -1 代表 unmatched
      - unmatch_cost:
          * 若 assign 中某 track unmatched，對 scalar cost 加上的固定成本

    內部流程：
      1) bbox cost（C_bbox）
         - 由 pso_bbox_cost() 計算，輸出 Tensor [M,N]
         - center_cost：
             * 取 bbox 中心點距離 dist
             * 以影像對角線 diag = sqrt(H^2+W^2) 正規化
             * C_center = dist / diag
         - scale_cost：
             * 以 bbox area 的 log-ratio 表示尺度差異
             * C_scale = | log(Ac/Ap) |
         - C_bbox = alpha * C_center + beta * C_scale

      2) confidence cost（C_conf）
         - 由 pso_conf_cost() 計算，輸出 Tensor [M,N]
         - 以 log-ratio 量化前後 confidence 差異：
             * C_conf = | log(conf_cur / conf_prev) |
         - 使用 eps clamp 避免 log(0)

      3) 合成總成本矩陣（C_total）
         - 線性加權：
             C_total = w_app*C_app + w_bbox*C_bbox + w_conf*C_conf
         - 形狀固定為 [M,N]，可直接餵給 Hungarian

      4)（可選）計算 scalar total_cost（供 PSO fitness 使用）
         - 若提供 assign：
             * 逐 track 累積對應的 C_total[i, j]
             * unmatched（j=-1）加 unmatch_cost
             * 若發生 collision（同一 detection 被多 track 指派）：
               加 1e6 重大懲罰（hard penalty）
         - 注意：此段會把 C_total 轉到 CPU numpy（通常 PSO/Hungarian 在 CPU）

    輸出：
      Dict[str, Any]：
        - "C_total": Tensor [M,N]
        - "C_app"  : Tensor [M,N]
        - "C_bbox" : Tensor [M,N]
        - "C_conf" : Tensor [M,N]
        - "total_cost": float（僅在 assign != None 時提供）

    設計重點：
      - C_total 供 Hungarian：穩定一對一 baseline
      - total_cost 供 PSO：將矩陣成本映射為「assignment 的 scalar fitness」
      - bbox 採用「位置（center）」+「尺度（area log-ratio）」的可解釋建模
      - confidence 用 log-ratio，對比例變動較敏感且尺度一致
      - collision hard penalty 確保搜尋過程偏好 one-to-one 合法解
"""

import torch
from typing import Any, Dict, List, Optional, Tuple 
DEVICE = "cuda"
@torch.no_grad()
def cost_appearance(assign, C_app, unmatch_cost):
    """
    assign: list[int], length M, each in [0..N-1] or -1
    C_app:  [M,N] appearance cost matrix (from -log(P))
    """
    cost = 0.0
    used = set()

    for i, j in enumerate(assign):
        if j == -1:
            cost += unmatch_cost
        else:
            if j in used:
                cost += 1e6  # hard penalty for collision
            else:
                cost += C_app[i, j]
                used.add(j)
    return cost

@torch.no_grad()
def bbox_cost(
    boxes_prev: List[List[float]],   # M x [x1,y1,x2,y2]
    boxes_cur:  List[List[float]],   # N x [x1,y1,x2,y2]
    input_hw: Tuple[int, int],
    alpha: float = 1.0,   # center weight
    beta: float  = 1.0,   # scale weight
) -> Dict[str, torch.Tensor]:
    """
    Return:
        {
          "C_center": [M,N] normalized center distance (dist/diag),
          "C_scale":  [M,N] |log(Ac/Ap)|,
          "C_bbox":   [M,N] alpha*C_center + beta*C_scale
        }
    """
    device = DEVICE
    if not torch.cuda.is_available():
        device = "cpu"

    M = len(boxes_prev)
    N = len(boxes_cur)

    if M == 0 or N == 0:
        z = torch.zeros((M, N), device=device)
        return {"C_center": z, "C_scale": z, "C_bbox": z}

    bp = torch.tensor(boxes_prev, device=device)  # [M,4]
    bc = torch.tensor(boxes_cur,  device=device)  # [N,4]

    
    # 1) center distance
    
    cp = 0.5 * (bp[:, :2] + bp[:, 2:])   # [M,2]
    cc = 0.5 * (bc[:, :2] + bc[:, 2:])   # [N,2]

    diff = cp[:, None, :] - cc[None, :, :]   # [M,N,2]
    dist = torch.norm(diff, dim=-1)          # [M,N]

    H, W = input_hw
    diag = (H * H + W * W) ** 0.5
    diag = max(float(diag), 1e-6)

    


    
    # 2) scale (area log-ratio)
    
    wp = (bp[:, 2] - bp[:, 0]).clamp(min=1.0)
    hp = (bp[:, 3] - bp[:, 1]).clamp(min=1.0)
    scale_p = torch.sqrt(wp * wp + hp * hp).clamp_min(1.0)   # wp/hp 是 prev box 尺度
    C_center = dist / scale_p.unsqueeze(1)  # [M,N]
    Ap = wp * hp                              # [M]


    wc = (bc[:, 2] - bc[:, 0]).clamp(min=1.0)
    hc = (bc[:, 3] - bc[:, 1]).clamp(min=1.0)
    Ac = wc * hc                              # [N]

    C_scale = torch.abs(torch.log((Ac[None, :] / Ap[:, None]).clamp(min=1e-6)))  # [M,N]

    
    # 3) combined bbox cost
    
    C_bbox = alpha * C_center + beta * C_scale
    return {"C_center": C_center, "C_scale": C_scale, "C_bbox": C_bbox}


@torch.no_grad()
def conf_cost(
    conf_prev: List[float],   # M
    conf_cur:  List[float],   # N
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Return:
        C_conf: [M, N]
    """
    device = DEVICE
    if not torch.cuda.is_available():
        device = "cpu"
    M = len(conf_prev)
    N = len(conf_cur)

    if M == 0 or N == 0:
        return torch.zeros((M, N), device=device)

    cp = torch.tensor(conf_prev, device=device).clamp(min=eps)  # [M]
    cc = torch.tensor(conf_cur,  device=device).clamp(min=eps)  # [N]

    C_conf = torch.abs(
        torch.log(cc[None, :] / cp[:, None])
    )  # [M,N]

    return C_conf

@torch.no_grad()
def cal_cost(
    *,
    # --- appearance ---
    C_app: torch.Tensor,                 # [M,N]  (-log P)

    # --- bbox ---
    boxes_prev: List[List[float]],        # M x [x1,y1,x2,y2]
    boxes_cur:  List[List[float]],        # N x [x1,y1,x2,y2]
    input_hw: Tuple[int, int],

    # --- confidence ---
    conf_prev: List[float],               # M
    conf_cur:  List[float],               # N

    # --- weights ---
    w_app:  float = 1.0,
    w_bbox: float = 0.3,
    w_conf: float = 0.2,

    # --- bbox params ---
    alpha: float = 1.0,   # center
    beta:  float = 0.5,   # scale

    # --- PSO / assignment (optional) ---
    assign: Optional[List[int]] = None,   # length M, det idx or -1
    unmatch_cost: float = 10.0,
) -> Dict[str, Any]:
    

    device = C_app.device
    M, N = C_app.shape

    
    # bbox cost
    
    bbox_out = bbox_cost(
    boxes_prev=boxes_prev,
    boxes_cur=boxes_cur,
    input_hw=input_hw,
    alpha=alpha,
    beta=beta,
    )
    C_center = bbox_out["C_center"].to(device)
    C_scale  = bbox_out["C_scale"].to(device)
    C_bbox   = bbox_out["C_bbox"].to(device)


    
    # confidence cost
    
    C_conf = conf_cost(
        conf_prev=conf_prev,
        conf_cur=conf_cur,
    ).to(device)

    
    # total matrix cost
    
    C_total = (
        w_app  * C_app +
        w_bbox * C_bbox +
        w_conf * C_conf
    )

    out: Dict[str, Any] = {
        "C_total": C_total,
        "C_app":   C_app,
        "C_bbox":  C_bbox,
        "C_center": C_center,
        "C_scale":  C_scale,
        "C_conf":  C_conf,
    }


    
    # PSO scalar fitness (optional)
    if assign is not None:
       
        C_np = C_total.detach().cpu().numpy()

        cost = 0.0
        used = set()
        for i, j in enumerate(assign):
            if j == -1:
                cost += unmatch_cost
            else:
                if j in used:
                    cost += 1e6
                else:
                    cost += C_np[i, j]
                    used.add(j)

        out["total_cost"] = float(cost)

    return out
