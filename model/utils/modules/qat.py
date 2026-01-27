from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.quantization as tq
from collections import OrderedDict
from . import encoderAndHead as encoderAndHead

class QATWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.quant = tq.QuantStub()
        self.model = model
        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

def _strip_prefix(sd, prefixes=("module.", "model.")):
    out = OrderedDict()
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out

def build_qat_model(
    ckpt_path: str,
    device: str = "cpu",
    backend: str = "fbgemm",
) -> nn.Module:
    device = torch.device(device)

    # FP32 base
    base = encoderAndHead.Model(
        in_channels=512,
        out_channels=512,
        warmup_epochs=10,
        proj_dim=128,
    ).to(device)

    
    from model.utils.trainingScr.run_training import disable_inplace
    base.apply(disable_inplace)

    
    model = QATWrapper(base).to(device)

    
    torch.backends.quantized.engine = backend  
    model.qconfig = tq.get_default_qat_qconfig(backend)
    tq.prepare_qat(model, inplace=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = _strip_prefix(state, prefixes=("module.", "model."))

    #補齊 model. 前綴
    fixed = OrderedDict()
    for k, v in state.items():
        if k.startswith(("quant.", "dequant.")):
            fixed[k] = v
        elif k.startswith("model."):
            fixed[k] = v
        else:
            fixed["model." + k] = v

    missing, unexpected = model.load_state_dict(fixed, strict=False)
    print("[build_qat_model] missing:", len(missing), "unexpected:", len(unexpected))
    if unexpected:
        print("[unexpected] show up to 20:", unexpected[:20])
    if missing:
        print("[missing] show up to 20:", missing[:20])


    model.eval()
    return model




def sanitize_qat_state_dict(
    state_dict: Dict[str, torch.Tensor],
    *,
    strip_prefix: str = "model.",
    drop_prefixes: Tuple[str, ...] = ("quant.",),
    drop_substrings: Tuple[str, ...] = ("fake_quant", "activation_post_process", "observer", "zero_point", "scale"),
    keep_non_tensor: bool = False,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    keep_exact = {
        "head.logit_scale",
        "head.logit_bias",
        "model.head.logit_scale",
        "model.head.logit_bias",
    }

    for k, v in state_dict.items():
        if k in keep_exact:
            kk = k[len(strip_prefix):] if strip_prefix and k.startswith(strip_prefix) else k
            out[kk] = v
            continue

        if (not torch.is_tensor(v)) and (not keep_non_tensor):
            continue

        if any(k.startswith(p) for p in drop_prefixes):
            continue

        if any(s in k for s in drop_substrings):
            continue

        if strip_prefix and k.startswith(strip_prefix):
            k = k[len(strip_prefix):]

        out[k] = v

    return out


def build_fp32_from_qat_ckpt(
    ckpt_path: str,
    device: str = "cuda",
    *,
    in_channels: int = 512,
    out_channels: int = 512,
    warmup_epochs: int = 10,
    proj_dim: int = 128,
) -> nn.Module:
    device = torch.device(device)

    
    model = encoderAndHead.Model(
        in_channels=in_channels,
        out_channels=out_channels,
        warmup_epochs=warmup_epochs,
        proj_dim=proj_dim,
    ).to(device).eval()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw_state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt


    clean_state = sanitize_qat_state_dict(raw_state)


    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    print(f"[load fp32-from-qat] missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0:
        print("[missing] show up to 20:", missing[:20])
    if len(unexpected) > 0:
        print("[unexpected] show up to 20:", unexpected[:20])

    return model

@torch.no_grad()
def export_onnx(model: nn.Module, out_onnx: str, device: str = "cuda"):
    device = torch.device(device)

    
    dummy = torch.randn(1, 512, 7, 7, device=device)

    torch.onnx.export(
        model,
        dummy,
        out_onnx,
        opset_version=13,
        input_names=["input"],
        output_names=["emb"],
        dynamic_axes={"input": {0: "N"}, "emb": {0: "N"}},
        do_constant_folding=True,
    )
    print(f"[OK] exported: {out_onnx}")

