# training/tracking/utils/modules/card.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributed as dist
"Residual mobile block"
class DSC(nn.Module):
    """Depthwise Separable Convolution (Improved Version)"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 stride=1,
                 padding=1,
                 bias=False,
                 ):

        super().__init__()
        
        hidden = in_channels // 2
        
        # Depthwise branch
        self.depth = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1,
                      stride=stride, padding=0,  bias=bias),

            nn.Conv2d(hidden, hidden, kernel_size=kernel_size,
                      stride=1, padding=2, bias=bias, groups=hidden),

            nn.Conv2d(hidden, out_channels, kernel_size=1,
                      stride=1, padding=0, bias=bias),
        )

        # Pointwise branch
        self.point = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, stride=stride, padding=0, bias=bias),
            nn.Conv2d(hidden, hidden, kernel_size=kernel_size,
                      stride=1, padding=2, groups=hidden, bias=bias),
            nn.Conv2d(hidden, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        
        self.act = nn.SiLU(inplace=False)
        self.act2 = nn.Hardswish(inplace=False)
        self.bn= nn.BatchNorm2d(out_channels)
       
    def forward(self, x, is_reinforce=False):
        d = self.depth(x)
        p = self.point(x)
        out = d + p
        out = self.bn(out)
        if is_reinforce: 
            out = self.act(out)
        else:
            out = self.act2(out)
        return out
    
class SEBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 reduction = 4):
        super().__init__()
        hidden = in_channels // reduction
        self.squeeze = nn.AdaptiveAvgPool2d(1)

        self.excitation = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=True),
            nn.ReLU(inplace=False),
            nn.Linear(hidden, in_channels, bias=True),
            nn.Hardsigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        squeeze = self.squeeze(x).view(b, c)
        excitation = self.excitation(squeeze).view(b,c ,1, 1)

        return x * excitation
    
class Shake2Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, alpha):
        ctx.alpha = alpha
        return alpha * x1 + (1 - alpha) * x2

    @staticmethod
    def backward(ctx, grad_out):
        beta = torch.rand_like(ctx.alpha)
        return beta * grad_out, (1 - beta) * grad_out, None
    


class Shake2(nn.Module):
    def forward(self, x1, x2):
        if not self.training:
            return 0.5 * x1 + 0.5 * x2

        alpha = torch.rand(1, device=x1.device)
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(alpha, src=0)  # rank0 的 alpha 同步到所有 rank
        return Shake2Fn.apply(x1, x2, alpha)


        
class RMB(nn.Module):
    def __init__(self, in_channels, out_channels, stride =1, warmup_epochs =0):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        dsc_out_channel = out_channels
        self.dsc_reinforce = DSC(in_channels, out_channels, stride=stride)
        self.dsc_normal    = DSC(in_channels, out_channels, stride=stride)
        self.se = SEBlock(in_channels=dsc_out_channel)
        self.shake_noise = Shake2()
        self.transition = nn.Sequential(
            nn.Conv2d(  in_channels=dsc_out_channel*2, 
                        out_channels = dsc_out_channel,
                        kernel_size=1,
                        stride=1 ),
            nn.SiLU(inplace=False)
        )

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch  
    

        
    def forward(self, x):
        x_f   = self.dsc_reinforce(x, is_reinforce=True)
        x_n_f = self.dsc_normal(x, is_reinforce=False)

        x_f = self.se(x_f)

        
        x_f_c = x_f.clone()
        x_n_f_c = x_n_f.clone()

        x_cat = torch.cat([x_f_c, x_n_f_c], dim=1)
        x_cat = self.transition(x_cat)

        alpha = 0.5 if self.current_epoch < self.warmup_epochs else torch.rand(1, device=x_f.device)
        fuse = alpha * x_f + (1 - alpha) * x_n_f


        
        out = self.shake_noise(x_cat.clone(), fuse.clone())

        return out

    
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=128, dropout=0.2, init_logit_scale=10.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim, bias=False),
            nn.LayerNorm(in_dim),
            nn.SiLU(inplace=False),
            nn.Dropout(p=dropout),
            nn.Linear(in_dim, proj_dim, bias=True)
        )
        # 用 log 參數化確保 scale 永遠為正
        self.logit_scale = nn.Parameter(torch.tensor(math.log(init_logit_scale), dtype=torch.float32), requires_grad=False)
        self.logit_bias  = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=False)


    def forward(self, x):
        z = self.net(x)
        z = F.normalize(z, dim=1)
        return z
    
    def similarity_prob(self, z1, z2):
        """
        z1, z2: [B, D] 或 [D]
        回傳: sigmoid(scale*cos + bias) -> 機率
        """
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        cos = (z1 * z2).sum(dim=-1)  # [-1, 1]
        scale = self.logit_scale.exp().clamp(1.0, 100.0)  # 防爆
        logit = scale * cos + self.logit_bias
        prob = torch.sigmoid(logit)
        return prob
    
    @torch.no_grad()
    def pairwise_prob_matrix(self, z: torch.Tensor) -> torch.Tensor:
        # z: [N,D] -> [N,N] sigmoid prob
        z = F.normalize(z, dim=1)
        cos = z @ z.t()
        scale = self.logit_scale.exp().clamp(1.0, 100.0)
        logits = scale * cos + self.logit_bias
        return torch.sigmoid(logits)

    @torch.no_grad()
    def relation_prob_matrix(self, z: torch.Tensor, tau: float=0.2, mask_diag: bool=True) -> torch.Tensor:
        # z: [N,D] -> [N,N] row-wise softmax prob
        z = F.normalize(z, dim=1)
        logits = (z @ z.t()) / tau
        if mask_diag:
            logits = logits.clone()
            logits.fill_diagonal_(float("-inf"))
        return torch.softmax(logits, dim=1)

        

    