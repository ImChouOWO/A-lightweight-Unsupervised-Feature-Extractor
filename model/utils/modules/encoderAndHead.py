# encoderAndHead.py
from pathlib import Path
import sys
import os
ROOT = Path(__file__).resolve().parents[3] 
sys.path.append(str(ROOT))
import torch
import torch.nn as nn
import torch.nn.functional as F
import model.utils.modules.card as card
class Model(nn.Module):
    def __init__(self,
                in_channels=None,
                out_channels=None,
                warmup_epochs = 10,
                proj_dim = 128):
        super().__init__()
        self.rmb = card.RMB(in_channels=in_channels, out_channels=out_channels, stride=1, warmup_epochs=warmup_epochs)
        self.head = card.ProjectionHead(in_dim=out_channels, proj_dim=proj_dim, dropout=0.2)
        
    def forward(self, x):
        # x: [B, C, H, W]
        feat = self.rmb(x)          # [B, C, H, W]
        feat = feat.mean(dim=(2, 3))     # GAP → [B, C]
        z = self.head(feat)              # [B, D]
        return z
    

