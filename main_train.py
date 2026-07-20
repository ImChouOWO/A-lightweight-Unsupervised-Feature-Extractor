from model.utils.trainingScr.run_training import _train
import sys
import os
import torch.distributed as dist

# torchrun --nproc_per_node=2 main_train.py
# Training script for training the model, it will generate a log file in the specified path with the results of the training.
if __name__ == "__main__":
    try:
        _train(path="model/conf")
    finally:
        # 確保 DDP 正確銷毀
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

        sys.exit(0)
