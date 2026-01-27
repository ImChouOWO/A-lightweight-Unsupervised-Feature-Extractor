from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
)
import os
import pickle
import numpy as np
import onnx
from tqdm import tqdm

import torch
from model.utils.modules.qat import build_fp32_from_qat_ckpt

def iter_roi_from_pkl(pkl_path):
    """yield torch.FloatTensor [512,10,10] one by one"""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    iterator = data if isinstance(data, (list, tuple)) else data.values()
    for sample in iterator:
        #roi_feats: torch.Tensor [N,512,10,10]
        if isinstance(sample, dict) and "roi_feats" in sample:
            roi = sample["roi_feats"]
        else:
            roi = sample

        if torch.is_tensor(roi):
            roi = roi.float()
        elif isinstance(roi, np.ndarray):
            roi = torch.from_numpy(roi).float()
        else:
            raise TypeError(type(roi))

        if roi.dim() == 3:
            yield roi
        elif roi.dim() == 4:
            for i in range(roi.size(0)):
                yield roi[i]
        else:
            raise RuntimeError(f"Unexpected roi shape: {tuple(roi.shape)}")


def export_fp32_onnx(ckpt_path, out_fp32_onnx, device="cpu"):
    device = torch.device(device)
    model = build_fp32_from_qat_ckpt(ckpt_path=ckpt_path, device=str(device)).eval()

    dummy = torch.randn(1, 512, 10, 10, device=device)
    torch.onnx.export(
        model,
        dummy,
        out_fp32_onnx,
        opset_version=13,
        do_constant_folding=True,
        input_names=["feat"],
        output_names=["emb"],
        dynamic_axes={"feat": {0: "B"}, "emb": {0: "B"}},
    )
    print(f"[OK] FP32 ONNX exported: {out_fp32_onnx}")

    m = onnx.load(out_fp32_onnx)
    onnx.checker.check_model(m)
    print("[OK] onnx.checker passed (fp32)")

class RoiCalibDataReader(CalibrationDataReader):
    def __init__(self, pkl_path, input_name="feat", batch_size=256, num_batches=200):
        self.pkl_path = pkl_path
        self.input_name = input_name
        self.batch_size = batch_size
        self.num_batches = num_batches

        self._iter = None
        self._done = False

    def get_next(self):
        if self._done:
            return None

        if self._iter is None:
            self._iter = iter_roi_from_pkl(self.pkl_path)
            self._pbar = tqdm(total=self.num_batches, desc="ORT calibration", ncols=100)

        buf = []
        try:
            while len(buf) < self.batch_size:
                roi = next(self._iter)  # [512,10,10]
                buf.append(roi.numpy().astype(np.float32))
        except StopIteration:
            
            if len(buf) == 0:
                self._done = True
                self._pbar.close()
                return None

        x = np.stack(buf, axis=0)  # [B,512,10,10]
        self._pbar.update(1)

        
        if self._pbar.n >= self.num_batches:
            self._done = True
            self._pbar.close()

        return {self.input_name: x}


def export_qdq_onnx(fp32_onnx, out_qdq_onnx, cali_pkl, batch_size=256, num_batches=200):
    dr = RoiCalibDataReader(
        pkl_path=cali_pkl,
        input_name="feat",
        batch_size=batch_size,
        num_batches=num_batches,
    )

    quantize_static(
        model_input=fp32_onnx,
        model_output=out_qdq_onnx,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,            # 產生 QuantizeLinear / DequantizeLinear
        activation_type=QuantType.QInt8,         # activation int8
        weight_type=QuantType.QInt8,             # weight int8
        per_channel=True,                        
        reduce_range=False,
        extra_options={
        "ActivationSymmetric": True,
        "WeightSymmetric": True,
        },
    )

    print(f"[OK] QDQ ONNX exported: {out_qdq_onnx}")

    m = onnx.load(out_qdq_onnx)
    onnx.checker.check_model(m)
    print("[OK] onnx.checker passed (qdq)")

    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    print("[QDQ] QuantizeLinear:", ops.get("QuantizeLinear", 0))
    print("[QDQ] DequantizeLinear:", ops.get("DequantizeLinear", 0))

if __name__ == "__main__":
    ckpt_path = "PATH/TO/qat_checkpoint.pth"  # Replace with the actual path to your QAT checkpoint
    cali_pkl  = "PATH/TO/cali_data.pkl"      # Replace with the actual path to your calibration data pickle

    fp32_onnx = "qat/onnx/encoder_fp32.onnx"
    qdq_onnx  = "encoder_qdq.onnx"

    export_fp32_onnx(ckpt_path, fp32_onnx, device="cpu")
    export_qdq_onnx(fp32_onnx, qdq_onnx, cali_pkl, batch_size=256, num_batches=200)
    print("[OK] ONNX export all done.")

    