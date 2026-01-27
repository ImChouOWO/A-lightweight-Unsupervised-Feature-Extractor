import sys
from pathlib import Path
import json
import torch

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())
import torch
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords, xyxy2xywh, check_img_size
import cv2
from models.experimental import attempt_load
from utils.torch_utils import select_device


class YoloDetects:

    def __init__(self,
                 weights,
                 conf_thres=0.4,
                 iou_thres=0.45,
                 img_size=1280,
                 device: str | None = None,):
        if device is None:
            self.device = select_device('')
        else:
            if device == "mps":
                self.device = torch.device("mps")
            elif device == "cuda":
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device("cpu")
        self.model = attempt_load(weights, map_location="cpu")
        self.model.to(self.device).eval()

        
        self.backbone_feat = None
        def _backbone_hook(module, input, output):
            # output: [B, C, Hf, Wf]
            self.backbone_feat = output
        for m in self.model.modules():
            if m.__class__.__name__ in ['SPPCSPC']:
                m.register_forward_hook(_backbone_hook)
                break
        self.half = (self.device.type == "cuda")
        if self.half:
            self.model.half()

        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        
        # 模型預熱
        dummy_input = torch.zeros(1, 3, img_size, img_size, device=self.device)
        if self.half:
            dummy_input = dummy_input.half()
        _ = self.model(dummy_input)


    
        self.stride = int(self.model.stride.max())
        self.img_size = check_img_size(img_size, s=self.stride)


    def run(self, frame):

        # stride = int(self.model.stride.max())
        self.img_size = check_img_size(self.img_size, s=self.stride)
        
        # 預處理
        img, orig_frame, _, _, _ = self._preprocess(frame)

        # 推論
        with torch.no_grad():
            pred = self.model(img, augment=False)[0]
            pred = non_max_suppression(pred, self.conf_thres,
                                       self.iou_thres)[0]

        result = []
        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4],
                                       frame.shape).round()
            for *xyxy, conf, cls in reversed(pred):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(
                    1, 4))).view(-1).tolist()
                cx, cy, w, h = xywh
                result.append({
                    "x": cx,
                    "y": cy,
                    "w": w,
                    "h": h,
                    "conf": float(conf)
                })
        return result
    
    def _preprocess(self, frame):
        """
        將 BGR frame 轉成 YOLO 所需的 tensor。
        return:
            img:         [1, 3, H, W] (already to device, normalize)
            orig_frame:  原始影像 (for scale_coords)
        """
        self.img_size = check_img_size(self.img_size, s=self.stride)

        img_lb, ratio, pad = letterbox(frame, new_shape=self.img_size, auto=False)  # 保留 ratio, pad
        input_hw = img_lb.shape[:2]  # (H_in, W_in)

        img = img_lb[:, :, ::-1].transpose(2, 0, 1).copy()
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img, frame, ratio, pad, input_hw
    
    def run_with_tensor(self, frame, return_img_tensor=False, cand_gate = 5):
        """
        return:
            result:      與 run() 相同的 bbox 結果 list[dict]
            pred_raw:    這張影像經過 YOLOv7 網路後的最終輸出張量 (NMS 前)
                         shape: [1, N, 5 + num_classes]，例如 [1, 25200, 85]
            (optional) img_tensor: 前處理後輸入模型的影像張量 [1, 3, H, W]
        """
        img, orig_frame, ratio, pad, input_hw = self._preprocess(frame)

        self.backbone_feat = None

        with torch.no_grad():
            pred_raw = self.model(img, augment=False)[0]
        # pred_raw shape: [1, N, 5+nc]
        obj = pred_raw[0, :, 4]  # objectness, shape [N]
        cand_count = int((obj > self.conf_thres).sum().item())

        if cand_count < cand_gate:
            pred_nms = None
            feat = None 
        else:
            pred_nms = non_max_suppression(pred_raw, self.conf_thres, self.iou_thres)[0]
            feat = self.backbone_feat 
        

        result = []

        if pred_nms is not None and len(pred_nms):
            # pred_nms_xyxy_in: 模型輸入座標(letterbox )用於 ROI
            pred_nms_xyxy_in = pred_nms[:, :4].clone()

            # pred_nms_xyxy_orig: 
            pred_nms[:, :4] = scale_coords(img.shape[2:], pred_nms[:, :4], orig_frame.shape).round()

            for i, (*xyxy_orig, conf, cls) in enumerate(reversed(pred_nms)):
                # 原圖座標 -> 原本的輸出格式
                xywh = (xyxy2xywh(torch.tensor(xyxy_orig).view(1, 4))).view(-1).tolist()
                cx, cy, w, h = xywh

                
                j = pred_nms.shape[0] - 1 - i
                xyxy_in = pred_nms_xyxy_in[j].detach().cpu().tolist()  # [x1,y1,x2,y2] in input coords

                result.append({
                    "x": cx, "y": cy, "w": w, "h": h,               # 原圖座標（不改）
                    "conf": float(conf),
                    "xyxy_in": xyxy_in,                             # ROI 用（模型輸入座標）
                    "input_hw": input_hw,                           # ROI 縮放用 (H_in,W_in)
                    "ratio": ratio,                                 # letterbox ratio
                    "pad": pad,                                     # letterbox pad (pad_w, pad_h)

                })

        if return_img_tensor:
            return result, pred_raw, feat
        else:
            return result, pred_raw
