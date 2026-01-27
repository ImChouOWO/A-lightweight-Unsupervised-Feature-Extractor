from pathlib import Path
import sys
import os
ROOT = Path(__file__).resolve().parents[3] 
sys.path.append(str(ROOT))
import torch
from torchvision.ops import roi_align
from torch.utils.data import Dataset
import model.yolov7.yoloDetects2 as yoloDet
import cv2
import pickle
from tqdm import tqdm



class PreProcess:
    def __init__(self, weight_path, dataset_path, preprocess_res_path, roi_patch_size=(10,10)):
        self.weight_path = weight_path
        self.yoloClass = yoloDet.YoloDetects(weights=self.weight_path)
        self.dataset_path = dataset_path
        self.preprocess_res_path = preprocess_res_path 
        self.roi_patch_size = roi_patch_size

    def _preprocess_roi(
        self,
        feat: torch.Tensor,        # [1,C,Hf,Wf]
        bboxes_xyxy: torch.Tensor, # [N,4] image coords
        img_hw: tuple,
        output_size=(10, 10),
        sampling_ratio=2,
        aligned=True,
        enforce_min_size=1.0,
    ):
        assert feat.dim() == 4 and feat.size(0) == 1, f"feat shape expected [1,C,H,W], got {feat.shape}"
        bboxes_xyxy = bboxes_xyxy.to(device=feat.device, dtype=torch.float32)

        B, C, Hf, Wf = feat.shape
        img_h, img_w = img_hw

        # add batch index
        batch_idx = torch.zeros((bboxes_xyxy.size(0), 1), device=bboxes_xyxy.device)
        boxes = torch.cat([batch_idx, bboxes_xyxy], dim=1).float()

        # sort coords
        x1 = torch.minimum(boxes[:, 1], boxes[:, 3])
        y1 = torch.minimum(boxes[:, 2], boxes[:, 4])
        x2 = torch.maximum(boxes[:, 1], boxes[:, 3])
        y2 = torch.maximum(boxes[:, 2], boxes[:, 4])
        boxes = torch.stack([boxes[:, 0], x1, y1, x2, y2], dim=1)

        # image → feature
        scale_x = Wf / float(img_w)
        scale_y = Hf / float(img_h)
        boxes[:, 1] *= scale_x
        boxes[:, 3] *= scale_x
        boxes[:, 2] *= scale_y
        boxes[:, 4] *= scale_y

        boxes[:, 1].clamp_(0, Wf - 1)
        boxes[:, 3].clamp_(0, Wf - 1)
        boxes[:, 2].clamp_(0, Hf - 1)
        boxes[:, 4].clamp_(0, Hf - 1)

        if enforce_min_size > 0:
            boxes[:, 3] = torch.maximum(boxes[:, 3], boxes[:, 1] + enforce_min_size)
            boxes[:, 4] = torch.maximum(boxes[:, 4], boxes[:, 2] + enforce_min_size)
            boxes[:, 3].clamp_(0, Wf - 1)
            boxes[:, 4].clamp_(0, Hf - 1)


        roi_feats = roi_align(
            input=feat,
            boxes=boxes,
            output_size=output_size,
            spatial_scale=1.0,
            sampling_ratio=sampling_ratio,
            aligned=aligned,
        )
        return roi_feats  # [N,C,H,W]

    def _preprocess_yolov7(self):
        imgs = os.listdir(self.dataset_path)
        datasets = []
        print("\033[1;37;42m[Hint] Starting Extracting YOLOv7 features...\033[0m")

        for name in tqdm(imgs, desc="Extract YOLOv7 features"):
            path = os.path.join(self.dataset_path, name)
            img = cv2.imread(path)
            if img is None:
                continue

            img_h, img_w = img.shape[:2]
            bbox_info, _, img_feat = self.yoloClass.run_with_tensor(
                img, return_img_tensor=True
            )
            if not bbox_info:
                continue

            # collect xyxy
            xyxy_list = []
            valid_boxes = []
            for box in bbox_info:
                if isinstance(box, dict) and "xyxy_in" in box:
                    xyxy_list.append(box["xyxy_in"])
                    valid_boxes.append(box)

            if len(xyxy_list) == 0:
                continue
            input_hw = bbox_info[0]["input_hw"]  # (H_in, W_in)   
            bboxes = torch.tensor(xyxy_list, dtype=torch.float32, device=img_feat.device)
            img_feat_f = img_feat.float()
            bboxes_f   = bboxes.float()

            roi_feats = self._preprocess_roi(
                feat=img_feat_f,
                bboxes_xyxy=bboxes_f,
                img_hw= input_hw,
                output_size=self.roi_patch_size,
                
            )

            res = {
                "img_name": name,
                "img_size": (img_h, img_w),
                "bbox_info": valid_boxes,          # 保留
                # "img_feat": img_feat_f.detach().cpu(),   # 存成 float32 CPU
                "roi_feats": roi_feats.detach().cpu(),   # 存成 float32 CPU
            }
            datasets.append(res)
            
        rank0_data = datasets[::2]
        rank1_data = datasets[1::2]

        print("\033[1;37;42m[Hint] Extracting done. Saving preprocess results...\033[0m")
        with open(f"{self.preprocess_res_path}/rank0.pkl", "wb") as f:
            pickle.dump(rank0_data, f)
        with open(f"{self.preprocess_res_path}/rank1.pkl", "wb") as f:
            pickle.dump(rank1_data, f)
        print(f"[OK] Preprocess done. Saved to {self.preprocess_res_path}")

    
class CreatDataset(Dataset):

    def __init__(self, pkl_path, min_conf=0.0, transform=None):
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)

        self.transform = transform
        self.roi_index = []
        


        for img_idx, sample in enumerate(self.data):
            rois = sample["roi_feats"]
            bboxes = sample["bbox_info"]
            if sample["roi_feats"].size(0) != len(sample["bbox_info"]):
                print(f"Warning: Mismatched roi_feats and bbox_info lengths in image {sample['img_name']}, skipping inconsistent boxes.")
                continue  
            for roi_idx, box in enumerate(bboxes):
                if isinstance(box, dict) and "conf" in box:
                    if box["conf"] < min_conf:
                        continue
                self.roi_index.append((img_idx, roi_idx))
        
        assert len(self.roi_index) > 0, "No valid ROI found"

    def __len__(self):
        return len(self.roi_index)

    def __getitem__(self, idx):
        img_idx, roi_idx = self.roi_index[idx]
        roi = self.data[img_idx]["roi_feats"][roi_idx]  # [C,H,W]

        v1 = roi.clone()
        v2 = roi.clone()
        
        if self.transform is not None:
            v1 = self.transform(v1)
            v2 = self.transform(v2)
            t = roi.mean(dim=(1,2))
        return v1, v2, t

    

class FeatureAugment:
    def __init__(self, noise_std=0.05, drop_prob=0.1):
        self.noise_std = noise_std
        self.drop_prob = drop_prob

    def __call__(self, x):
        # x: [C,H,W]
        if torch.rand(1) < self.drop_prob:
            x = x * (torch.rand_like(x) > 0.2)
        x = x + torch.randn_like(x) * self.noise_std
        return x


        
    
        
            


        
        



