import subprocess
from pathlib import Path

TRTEXEC = "PATH/TO/trtexec"  # Replace with the actual path to trtexec
ROOT = Path(__file__).resolve().parent
MINBATCH = 1
OPTBATCH = 16
MAXBATCH = 32
onnx_path = ROOT / "qat/onnx/encoder_fp32.onnx"
engine_path = ROOT / "qat/tensorRT/encoder_int8.engine"
log_path = ROOT / "qat/build_log/trtexec_build.log"

cmd = [
    TRTEXEC,
    f"--onnx={onnx_path}",
    "--int8",
    "--fp16",
    f"--minShapes=feat:{MINBATCH}x512x10x10",
    f"--optShapes=feat:{OPTBATCH}x512x10x10",
    f"--maxShapes=feat:{MAXBATCH}x512x10x10",
    f"--saveEngine={engine_path}",
    "--profilingVerbosity=detailed",
    "--verbose",
]


with log_path.open("w") as f:
    subprocess.run(
        cmd,
        check=True,
        stdout=f,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT)
    )



print(f"[OK] engine: {engine_path}")
print(f"[OK] log: {log_path}")

cmd = [
    TRTEXEC,
    f"--loadEngine={engine_path}",
    "--dumpLayerInfo",
    "--profilingVerbosity=detailed",
    "--verbose",
]
    
subprocess.run(
    cmd
)

