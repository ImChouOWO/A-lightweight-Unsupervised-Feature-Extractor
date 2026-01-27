import time
import threading
import psutil
import cv2
import numpy as np

try:
    import pynvml
    _NVML_OK = True
except Exception:
    _NVML_OK = False


class ResourceMonitor:
    def __init__(self, gpu_index: int = 0, sample_interval: float = 0.2):
        self.gpu_index = gpu_index
        self.sample_interval = float(sample_interval)

        self.cpu_util = 0.0
        self.ram_used_gb = 0.0
        self.ram_total_gb = 0.0

        self.gpu_util = 0.0
        self.vram_used_gb = 0.0
        self.vram_total_gb = 0.0

        self._stop = threading.Event()
        self._th = None
        self._proc = psutil.Process()

        self._nvml_handle = None
        if _NVML_OK:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            except Exception:
                self._nvml_handle = None

    def start(self):
        # prime cpu_percent measurement
        psutil.cpu_percent(interval=None)
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def close(self):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=1.0)
        # nvml shutdown (optional)
        if _NVML_OK:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.cpu_util = float(psutil.cpu_percent(interval=None))
                vm = psutil.virtual_memory()
                self.ram_used_gb = float((vm.total - vm.available) / (1024 ** 3))
                self.ram_total_gb = float(vm.total / (1024 ** 3))
            except Exception:
                pass

            if self._nvml_handle is not None:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                    self.gpu_util = float(util.gpu)
                    self.vram_used_gb = float(mem.used / (1024 ** 3))
                    self.vram_total_gb = float(mem.total / (1024 ** 3))
                except Exception:
                    # NVML failure fallback to zeros
                    self.gpu_util = 0.0
                    self.vram_used_gb = 0.0
                    self.vram_total_gb = 0.0

            time.sleep(self.sample_interval)

class ImgAug:
    def __init__(self):
        pass
    def mosaic_4(self, images, out_size=1280):
        """
        images: list of 4 images (H,W,3)
        labels: list of list, each label is [cls, cx, cy, w, h] (normalized)
        """
        assert len(images) == 4

        h, w = out_size, out_size
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # mosaic center
        xc, yc = w // 2, h // 2

        placements = [
            (0, 0, xc, yc),        # top-left
            (xc, 0, w, yc),        # top-right
            (0, yc, xc, h),        # bottom-left
            (xc, yc, w, h),        # bottom-right
        ]

        for i, img in enumerate(images):
            x1, y1, x2, y2 = placements[i]

            resized = cv2.resize(img, (x2 - x1, y2 - y1))
            canvas[y1:y2, x1:x2] = resized
        

        return canvas
    def flip(self, img):
        vertical = cv2.flip(img, 0)
        horizontal = cv2.flip(img, 1)
        return vertical, horizontal
    

    def _do_augmentation(self, img_dir: str, save_path: str):
        import os
        import cv2
        import numpy as np
        from tqdm import tqdm

        
        os.makedirs(save_path, exist_ok=True)
        flip_dir = os.path.join(save_path, "flip")
        os.makedirs(flip_dir, exist_ok=True)
        imgs = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
        ])

        print("[Hint] Do flip augmentation")

        # ---------- Flip ----------
        with tqdm(total=len(imgs), desc="Images flip", unit="img", dynamic_ncols=True) as pbar:
            for name in imgs:
                img_path = os.path.join(img_dir, name)
                img = cv2.imread(img_path)

                if img is None:
                    pbar.update(1)
                    continue

                v_img, h_img = self.flip(img)

                base, ext = os.path.splitext(name)

                cv2.imwrite(
                    os.path.join(flip_dir, f"{base}_vflip{ext}"),
                    v_img
                )
                cv2.imwrite(
                    os.path.join(flip_dir, f"{base}_hflip{ext}"),
                    h_img
                )

                pbar.update(1)

        # ---------- Mosaic ----------
        print("[Hint] Do mosaic augmentation")

        mosaic_dir = os.path.join(save_path, "mosaic")
        os.makedirs(mosaic_dir, exist_ok=True)

        # 每 4 張做一次 mosaic
        with tqdm(total=len(imgs) // 4, desc="Mosaic", unit="group", dynamic_ncols=True) as pbar:
            for i in range(0, len(imgs) - 3, 4):
                group = imgs[i:i + 4]
                images = []

                for name in group:
                    img = cv2.imread(os.path.join(img_dir, name))
                    if img is not None:
                        images.append(img)

                if len(images) != 4:
                    pbar.update(1)
                    continue

                mosaic = self.mosaic_4(images)

                mosaic_name = f"mosaic_{i:06d}.jpg"
                cv2.imwrite(
                    os.path.join(mosaic_dir, mosaic_name),
                    mosaic
                )

                pbar.update(1)
        print("[Hint] Image augmentation done")
            
if __name__ == "__main__":
    aug = ImgAug()
    aug._do_augmentation(
        img_dir="/home/soic/Desktop/tracking/v24/training/tracking/dataset/train",
        save_path="/home/soic/Desktop/tracking/v24/training/tracking/dataset/aug"
    )