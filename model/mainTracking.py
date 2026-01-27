from model.utils.inferScr.infer import MainInfer
from  model.utils.costTool.costCard import cal_cost
import model.utils.costTool.KalmanFilter as KF
from  model.utils.costTool.hung import hungarian_assign
from dataclasses import dataclass, field
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import torch
from filterpy.kalman import KalmanFilter
import yaml
def load_conf(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@dataclass
class TrackMemory:
    encoder_feat: Optional[np.ndarray] = None        # 128D np.float32

    feat_historical: List[np.ndarray] = field(default_factory=list)
    conf_historical: List[float] = field(default_factory=list)
    bbox_historical: List[Tuple[float, float, float, float]] = field(default_factory=list)

    last_conf: Optional[float] = None
    last_update_frame: Optional[int] = None

    lat_AIS: Optional[Dict[str, Any]] = None
    last_bbox: Optional[Tuple[float, float, float, float]] = None

    age: int = 0
    misss_count: int = 0
    last_match_cost: Optional[float] = None
    state: Optional[str] = None


@dataclass
class TrackState:
    track_id: int
    kf: KalmanFilter
    memory: TrackMemory
    age: int = 1
    miss_count: int = 0
    state: str = "ACTIVE"      # "ACTIVE" or "LOST"


class Tracking:
    def __init__(self):
        conf = load_conf(path="model/conf/conf.yaml")

        #  tracker config block 
        if "tracker" not in conf:
            raise KeyError("Missing 'tracker' section in YAML config.")
        tcfg = conf["tracker"]

        # thresholds / config
        self.init_conf_min = float(tcfg.get("init_conf_min", 0.5))
        self.hist_max      = int(tcfg.get("hist_max", 10))
        self.emb_top_k     = int(tcfg.get("emb_top_k", 5))

        # runtime state (NOT in YAML)
        self.tracks: Dict[int, TrackState] = {}
        self.next_id: int = 0

        # appearance (numerical stability / temperature)
        # YAML uses app_tau; class uses self.tau
        self.tau = float(tcfg.get("app_tau", 0.07))
        self.eps = float(tcfg.get("eps", 1e-12))

        # cost weights
        self.w_app = float(tcfg.get("w_app", 1.0))
        self.w_bbox = float(tcfg.get("w_bbox", 0.3))
        self.w_conf = float(tcfg.get("w_conf", 0.2))
        self.alpha = float(tcfg.get("alpha", 1.0))
        self.beta = float(tcfg.get("beta", 0.5))

        # unmatched cost
        self.unmatch_cost = float(tcfg.get("unmatch_cost", 10.0))

        # association gates
        self.cost_max = float(tcfg.get("cost_max", 50.0))
        self.max_age  = int(tcfg.get("max_age", 30))

        # update_matched gates
        self.ema_alpha       = float(tcfg.get("ema_alpha", 0.9))
        self.conf_update_min = float(tcfg.get("conf_update_min", 0.55))
        self.cost_update_max = float(tcfg.get("cost_update_max", 30.0))
        self.maha_thr        = float(tcfg.get("maha_thr", 9.49))

        # long-lost handling: ReID-only stage
        self.lost_reid_after = int(tcfg.get("lost_reid_after", 60))
        self.reid_sim_min    = float(tcfg.get("reid_sim_min", 0.6))

        # YAML may already provide reid_only_cost_max (=1-reid_sim_min). If not, derive it.
        if "reid_only_cost_max" in tcfg:
            self.reid_only_cost_max = float(tcfg["reid_only_cost_max"])
        else:
            self.reid_only_cost_max = 1.0 - self.reid_sim_min


    def creat_item(self, emb, conf, bbox, frame) -> Optional[TrackMemory]:
        """
        Create a new TrackMemory from one detection (emb/conf/bbox) at frame.
        """
     
        if conf is None or float(conf) < float(self.init_conf_min):
            return None

       
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        if emb.shape[0] != 128:
            raise ValueError(f"emb must be shape (128,), got {emb.shape}")

    
        norm = float(np.linalg.norm(emb) + 1e-12)
        emb = emb / norm

        tmp = TrackMemory(
            encoder_feat=emb,
            last_conf=float(conf),
            last_bbox=tuple(map(float, bbox)),
            last_update_frame=int(frame),
            lat_AIS=None,
            age=1,
            misss_count=0,
            last_match_cost=None,
            state="ACTIVE",
        )

       
        tmp.feat_historical.append(emb)
        tmp.conf_historical.append(float(conf))
        tmp.bbox_historical.append(tuple(map(float, bbox)))

        # 限制 pool 長度
        if len(tmp.feat_historical) > self.hist_max:
            tmp.feat_historical = tmp.feat_historical[-self.hist_max:]
            tmp.conf_historical = tmp.conf_historical[-self.hist_max:]
            tmp.bbox_historical = tmp.bbox_historical[-self.hist_max:]

        return tmp
    @torch.no_grad()
    def build_C_app_topk(
        self,
        *,
        row_to_tid: List[int],          # M
        det_embs: List[np.ndarray],     # N x (128,)
        device: Optional[str] = None,
        topk: int = 5,
        use_topk_mean: bool = True,    
        fallback_to_ema: bool = True,   
    ) -> torch.Tensor:
        """
        Return:
        C_app: [M,N], smaller is better.
        C_app = 1 - sim_topk
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        M = len(row_to_tid)
        N = len(det_embs)

        if M == 0 or N == 0:
            return torch.zeros((M, N), device=device)

    
        det = np.stack([np.asarray(e, dtype=np.float32).reshape(-1) for e in det_embs], axis=0)
        det = np.nan_to_num(det, nan=0.0, posinf=0.0, neginf=0.0)
        det = det / (np.linalg.norm(det, axis=1, keepdims=True) + 1e-12)
        F_det = torch.from_numpy(det).to(device)          # [N,128]
        F_det = torch.nan_to_num(F_det, nan=0.0, posinf=0.0, neginf=0.0)
        C_app_rows = []

        for tid in row_to_tid:
            ts = self.tracks[tid]
            mem = ts.memory

            # list[np.ndarray] -> [T,128]
            bank_list = mem.feat_historical

            if (bank_list is None) or (len(bank_list) == 0):
                if fallback_to_ema and (mem.encoder_feat is not None):
                    bank_list = [mem.encoder_feat]
                else:
                    # 沒有外觀資訊：建立一個內容全為 1 的 cost row
                    C_app_rows.append(torch.ones((N,), device=device))
                    continue

            bank = np.stack([np.asarray(f, dtype=np.float32).reshape(-1) for f in bank_list], axis=0)
            bank = np.nan_to_num(bank, nan=0.0, posinf=0.0, neginf=0.0)
            bank = bank / (np.linalg.norm(bank, axis=1, keepdims=True) + 1e-12)
            F_bank = torch.from_numpy(bank).to(device)    # [T,128]
            F_bank = torch.nan_to_num(F_bank, nan=0.0, posinf=0.0, neginf=0.0)

            # cosine sim: [T,N]
            sim_TN = F_bank @ F_det.T
            sim_TN = torch.nan_to_num(sim_TN, nan=-1.0, posinf=1.0, neginf=-1.0)
            # Top-K
            k = min(int(topk), sim_TN.shape[0])
            if k <= 0:
                C_app_rows.append(torch.ones((N,), device=device))
                continue

            if use_topk_mean:
                topv, _ = torch.topk(sim_TN, k=k, dim=0)     # [k,N]
                sim = topv.mean(dim=0)                       # [N]
            else:
                sim = sim_TN.max(dim=0).values               # [N]

            #  cost = 1 - sim   (sim in [-1,1], cost in [0,2])
            C_app_rows.append(1.0 - sim)

        C_app = torch.stack(C_app_rows, dim=0)             # [M,N]
        C_app = torch.nan_to_num(C_app, nan=1.0, posinf=1.0, neginf=1.0)
        return C_app
    
    @torch.no_grad()
    def cal_cost(
        self,
        *,
        row_to_tid: List[int],
        det_embs: List[np.ndarray],                       # N 個 128D（np.float32, (128,)）
        det_boxes: List[List[float]],                     # N x [x1,y1,x2,y2]
        det_confs: List[float],                           # N
        input_hw: Tuple[int, int],                        # (H, W)
        device: Optional[str] = None,                     # "cuda" / "cpu"
        assign: Optional[List[int]] = None,               # 預開接口 debug 用，可不給
    ) -> Dict[str, Any]:

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        #  gather prev (tracks) 
        boxes_prev: List[List[float]] = []
        conf_prev: List[float] = []
        feat_prev: List[np.ndarray] = []

        for tid in row_to_tid:
            ts = self.tracks[tid]
            mem = ts.memory

            if mem.encoder_feat is None or mem.last_bbox is None or mem.last_conf is None:
                continue

            feat_prev.append(mem.encoder_feat.astype(np.float32, copy=False))
            boxes_prev.append(list(map(float, mem.last_bbox)))
            conf_prev.append(float(mem.last_conf))

        M = len(feat_prev)
        N = len(det_embs)

        if M == 0 or N == 0:
            C_app = torch.zeros((M, N), device=device)
            return cal_cost(
                C_app=C_app,
                boxes_prev=boxes_prev,
                boxes_cur=det_boxes,
                input_hw=input_hw,
                conf_prev=conf_prev,
                conf_cur=det_confs,
                w_app=self.w_app,
                w_bbox=self.w_bbox,
                w_conf=self.w_conf,
                alpha=self.alpha,
                beta=self.beta,
                assign=assign,
                unmatch_cost=self.unmatch_cost,
            )

        emb_cur = np.stack([np.asarray(e, dtype=np.float32).reshape(-1) for e in det_embs], axis=0)
        if emb_cur.shape[1] != 128:
            raise ValueError(f"det_embs must be 128D, got {emb_cur.shape}")
        emb_cur = np.nan_to_num(emb_cur, nan=0.0, posinf=0.0, neginf=0.0)
        emb_cur_norm = emb_cur / (np.linalg.norm(emb_cur, axis=1, keepdims=True) + self.eps)

        emb_prev = np.stack([e.reshape(-1) for e in feat_prev], axis=0)
        emb_prev_norm = emb_prev / (np.linalg.norm(emb_prev, axis=1, keepdims=True) + self.eps)

        F_prev = torch.from_numpy(emb_prev_norm).to(device)  # [M,128]
        F_cur  = torch.from_numpy(emb_cur_norm).to(device)   # [N,128]

        sim = F_prev @ F_cur.T
        P = torch.softmax(sim / float(self.tau), dim=1)  # [M,N]
        C_app = self.build_C_app_topk(
            row_to_tid=row_to_tid,
            det_embs=det_embs,
            device=device,
            topk=self.emb_top_k,  
            use_topk_mean=True,
        )


        out = cal_cost(
            C_app=C_app,
            boxes_prev=boxes_prev,
            boxes_cur=det_boxes,
            input_hw=input_hw,
            conf_prev=conf_prev,
            conf_cur=det_confs,
            w_app=self.w_app,
            w_bbox=self.w_bbox,
            w_conf=self.w_conf,
            alpha=self.alpha,
            beta=self.beta,
            assign=assign,
            unmatch_cost=self.unmatch_cost,
        )
        return out


    def apply_kalman_gating(
        self,
        C_total_np: np.ndarray,     # [M, N] numpy cost matrix
        row_to_tid: List[int],      # length M, row i -> track_id
        det_boxes: List[List[float]],  # length N, each [x1,y1,x2,y2]
        *,
        maha_thr: float = 13.28,    # chi-square(4 dof, 0.99) -> 13.28
        INF: float = 1e9,
    ) -> np.ndarray:
        """
        Apply Kalman gating BEFORE Hungarian:
        if Mahalanobis distance d^2 > maha_thr => C_total_np[i,j] = INF
        Return the gated cost matrix (in-place or copied).
        """
        M, N = C_total_np.shape
        if M == 0 or N == 0:
            return C_total_np

        
        C = C_total_np

        for i in range(M):
            tid = row_to_tid[i]
            ts = self.tracks[tid]
            kf = ts.kf  

            for j in range(N):
                bbox = det_boxes[j]
                d2 = KF.gating_distance_maha(kf, bbox) 
                if float(d2) > float(maha_thr):
                    C[i, j] = float(INF)

        return C

    def predict_all(self):
        # bbox_cost 
        for ts in self.tracks.values():
            ts.kf.predict()
            pred_bbox = KF.x_to_bbox_xyxy(ts.kf.x.reshape(-1))
            ts.memory.last_bbox = tuple(map(float, pred_bbox))

    def mark_missed(self, track_ids: List[int]):
        for tid in track_ids:
            ts = self.tracks.get(tid)
            if ts is None:
                continue
            ts.miss_count += 1
            ts.state = "LOST"
            ts.memory.misss_count = ts.miss_count
            ts.memory.state = "LOST"

    def purge_dead(self):
        dead = [tid for tid, ts in self.tracks.items() if ts.miss_count > self.max_age]
        for tid in dead:
            del self.tracks[tid]

    def create_new_tracks(self, det_ids, det_embs, det_boxes, det_confs, frame_id):
        for j in det_ids:
            conf = float(det_confs[j])
            if conf < self.init_conf_min:
                continue
            mem = self.creat_item(det_embs[j], conf, det_boxes[j], frame_id)
            if mem is None:
                continue
            kf = KF.init_kf_from_bbox(det_boxes[j])
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = TrackState(track_id=tid, kf=kf, memory=mem)

    def update_matched(
        self,
        matches,
        row_to_tid,
        det_embs,
        det_boxes,
        det_confs,
        frame_id,
        C_total_np,
        *,
        ema_alpha=0.9,
        conf_update_min=0.55,
        cost_update_max=50.0,
        maha_thr=9.49,   # chi-square(4 dof, 0.95) -> 9.49
    ):
        for row_i, det_j in matches:
            tid = row_to_tid[row_i]
            ts = self.tracks[tid]
            mem = ts.memory

            bbox = det_boxes[det_j]
            conf = float(det_confs[det_j])
            emb  = det_embs[det_j]  # np.ndarray (128,)

            # KF update
            ts.kf.update(KF.bbox_xyxy_to_z(bbox))

        
            mem.last_bbox = tuple(map(float, bbox))
            mem.last_conf = conf
            mem.last_update_frame = int(frame_id)
            mem.age += 1
            mem.misss_count = 0
            mem.state = "ACTIVE"

            ts.age += 1
            ts.miss_count = 0
            ts.state = "ACTIVE"

            cost_ij = float(C_total_np[row_i, det_j])
            mem.last_match_cost = cost_ij

            # 外觀更新 gate
            if conf < conf_update_min:
                continue
            if cost_ij > cost_update_max:
                continue

            
            d2 = KF.gating_distance_maha(ts.kf, bbox)
            if d2 > maha_thr:
                continue

            
            emb = np.asarray(emb, dtype=np.float32).reshape(-1)
            emb = emb / (np.linalg.norm(emb) + 1e-12)
            det_emb_norm = emb.copy()

            if mem.encoder_feat is None:
                mem.encoder_feat = det_emb_norm
            else:
                f = (ema_alpha * mem.encoder_feat + (1.0 - ema_alpha) * det_emb_norm).astype(np.float32)
                f = f / (np.linalg.norm(f) + 1e-12)
                mem.encoder_feat = f

            mem.feat_historical.append(det_emb_norm)
            mem.conf_historical.append(conf)
            mem.bbox_historical.append(mem.last_bbox)

            # 限制 pool 長度
            if len(mem.feat_historical) > self.hist_max:
                mem.feat_historical = mem.feat_historical[-self.hist_max:]
                mem.conf_historical = mem.conf_historical[-self.hist_max:]
                mem.bbox_historical = mem.bbox_historical[-self.hist_max:]

    def update(self, obj: Dict):
        det_embs  = obj.get("embs", []) or []
        det_boxes = obj.get("bboxes", []) or []
        det_confs = obj.get("confs", []) or []
        input_hw  = obj.get("input_hw", None)
        frame_id  = obj.get("frame_id", None)
        
        if input_hw is None:
            raise ValueError("obj['input_hw'] is required")
        if frame_id is None:
            raise ValueError("obj['frame_id'] is required")
        if not (len(det_embs) == len(det_boxes) == len(det_confs)):
            raise ValueError("Length mismatch: embs/bboxes/confs must have same length")

        N = len(det_boxes)

    
        if N == 0:
            all_ids = list(self.tracks.keys())
            self.mark_missed(all_ids)
            self.purge_dead()
            return [], all_ids, []

     
        if len(self.tracks) > 0:
            self.predict_all()

        
        rows_main = []
        rows_reid = []
        for tid, ts in self.tracks.items():
            if ts.miss_count > self.lost_reid_after:
                rows_reid.append(tid)   
            else:
                rows_main.append(tid)   

        rows_main = sorted(rows_main)
        rows_reid = sorted(rows_reid)

        all_matches = []
        unmatched_dets = list(range(N))

        
        unmatched_rows_main = list(range(len(rows_main)))  # default

        if len(rows_main) > 0:
            cost_out = self.cal_cost(
                row_to_tid=rows_main,
                det_embs=det_embs,
                det_boxes=det_boxes,
                det_confs=det_confs,
                input_hw=input_hw,
            )
            C_total_np = cost_out["C_total"].detach().cpu().numpy()
            

            # motion gating
            C_total_np = self.apply_kalman_gating(
                C_total_np,
                row_to_tid=rows_main,
                det_boxes=det_boxes,
                maha_thr=self.maha_thr,
                INF=1e9,
            )

            matches1, unmatched_rows_main, unmatched_dets = hungarian_assign(
                C_total_np,
                cost_max=self.cost_max,
            )

            # 更新 matched（ACTIVE / re-activate）
            self.update_matched(
                matches1,
                rows_main,
                det_embs,
                det_boxes,
                det_confs,
                frame_id,
                C_total_np,
                ema_alpha=self.ema_alpha,
                conf_update_min=self.conf_update_min,
                cost_update_max=self.cost_update_max,
                maha_thr=self.maha_thr,
            )
           
            all_matches.extend([(rows_main[r], d) for r, d in matches1]) 

       
            unmatched_track_ids_main = [rows_main[r] for r in unmatched_rows_main]
            self.mark_missed(unmatched_track_ids_main)
        else:
            unmatched_track_ids_main = []

      
        
     
        unmatched_track_ids_reid = []
        if len(rows_reid) > 0 and len(unmatched_dets) > 0:
            det_embs_u  = [det_embs[j] for j in unmatched_dets]
            det_boxes_u = [det_boxes[j] for j in unmatched_dets]
            det_confs_u = [det_confs[j] for j in unmatched_dets]

           
            C_app = self.build_C_app_topk(
                row_to_tid=rows_reid,
                det_embs=det_embs_u,
                topk=self.emb_top_k,
                use_topk_mean=True,
            )  # [M2, Nu]

            C_reid_np = C_app.detach().cpu().numpy()

            matches2, unmatched_rows_reid, unmatched_dets_u = hungarian_assign(
                C_reid_np,
                cost_max=self.reid_only_cost_max,
            )


            matches2_global = []
            for r2, du in matches2:
                tid = rows_reid[r2]
                det_id = unmatched_dets[du]
                matches2_global.append((tid, det_id))


            self.update_matched(
                matches2,
                rows_reid,
                det_embs_u,
                det_boxes_u,
                det_confs_u,
                frame_id,
                C_reid_np,
                ema_alpha=self.ema_alpha,
                conf_update_min=self.conf_update_min,
                cost_update_max=self.reid_only_cost_max,  
                maha_thr=1e18, 
            )

            all_matches.extend(matches2_global)

     
            unmatched_track_ids_reid = [rows_reid[r] for r in unmatched_rows_reid]
            self.mark_missed(unmatched_track_ids_reid)

           
            unmatched_dets = [unmatched_dets[du] for du in unmatched_dets_u]
        else:
            if len(rows_reid) > 0:
                self.mark_missed(rows_reid)
                unmatched_track_ids_reid = rows_reid

        # unmatched dets -> create new tracks
        self.create_new_tracks(unmatched_dets, det_embs, det_boxes, det_confs, frame_id)

        # delete dead tracks
        self.purge_dead()

        # return matches（tid, det_id）、unmatched track ids、unmatched det ids
        unmatched_track_ids = unmatched_track_ids_main + unmatched_track_ids_reid
        matches_tid = [(int(tid), int(det_j)) for tid, det_j in all_matches]
        return matches_tid, unmatched_track_ids, unmatched_dets



