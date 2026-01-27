import sys
from pathlib import Path

import torch

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())
import numpy as np
from pathlib import Path
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.datasets import letterbox
from utils.torch_utils import select_device


class YoloDetects:

    def __init__(self, weights, conf_thres=0.05, iou_thres=0.45, device=0,img_size=640):
        # 裝置與半精度設定
        self.device = select_device('')
        self.half = self.device.type != 'cpu'

        # 載入模型
        self.model = attempt_load(weights, map_location=self.device)
        if self.half:
            self.model.half()
        self.model.eval()
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size=img_size

    def run(self, img0):
        # 預處理
        # img0 = frame.copy()
        img_size = self.img_size
        img = letterbox(img0, new_shape=img_size)[0]
        # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # 加上 .copy()

        img = torch.from_numpy(img).to(self.device).float()
        img = img.half() if self.half else img
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        locations = []

        # 推論
        try:
            with torch.no_grad():
                pred = self.model(img, augment=False)[0]
                pred = non_max_suppression(pred, self.conf_thres,
                                           self.iou_thres)
        except Exception as e:
            print(f"[錯誤] 推論失敗：{e}")
            return locations

        for det in pred:
            if det is not None and len(det):
                # 將 bbox 從模型輸出尺寸對應到原圖
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4],
                                          img0.shape).round()

                detections = []
                det_wh_map = {}
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = map(int, xyxy)

                    # 中心點與尺寸
                    xywh = xyxy2xywh(
                        torch.tensor([x1, y1, x2,
                                      y2]).view(1, 4)).view(-1).tolist()
                    cx, cy, w, h = xywh
                    detections.append([cx, cy])
                    det_wh_map[(int(cx), int(cy))] = (w, h)
                    locations.append({
                        "x": cx,
                        "y": cy,
                        "w": w,
                        "h": h,
                        "conf": float(conf)
                    })
        return locations

    def updateconfig(self,conf_thres):
        self.conf_thres=conf_thres


if __name__ == '__main__':
    import cv2
    yoClass = YoloDetects(
        weights=
        r"D:\Desktop\v24_20250603\controller\utils\dataSensor\yolov7\yolov7_best.pt"
    )
    cap = cv2.VideoCapture('rtsp://admin:Soic123456@192.168.1.16:554/stream0')
    # startFrame = time.time()
    while (True):
        # 從攝影機擷取一張影像
        ret, frame = cap.read()
        if not ret:
            break
        locations = yoClass.detect(frame)
        print(locations)
