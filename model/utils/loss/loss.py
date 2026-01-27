import torch
import torch.nn as nn
import torch.nn.functional as F


class KLSimilarityLoss(nn.Module):
    """
    KL( P_teacher || P_student )
    P = softmax(sim / tau) over batch (exclude self)
    """
    def __init__(
        self,
        tau_t: float = 0.07,
        tau_s: float = 0.2,
        eps: float = 1e-8,
        reduction: str = "batchmean",
        neg_large: float = 1e9,  # avoid -inf for backend compatibility
    ):
        super().__init__()
        self.tau_t = float(tau_t)
        self.tau_s = float(tau_s)
        self.eps = float(eps)
        self.reduction = reduction
        self.neg_large = float(neg_large)

    def forward(self, teacher_feat: torch.Tensor, student_feat: torch.Tensor) -> torch.Tensor:
        assert teacher_feat.dim() == 2 and student_feat.dim() == 2
        B = teacher_feat.size(0)
        if B < 2:
            return teacher_feat.new_tensor(0.0)

        # ensure teacher is no-grad source (safe even if caller forgot detach)
        teacher_feat = teacher_feat.detach()

        t = F.normalize(teacher_feat, dim=1)
        s = F.normalize(student_feat, dim=1)

        sim_t = t @ t.t()  # [B,B]
        sim_s = s @ s.t()  # [B,B]

        # mask diagonal without in-place and without -inf
        mask = torch.eye(B, device=sim_t.device, dtype=torch.bool)
        sim_t = sim_t.masked_fill(mask, -self.neg_large)
        sim_s = sim_s.masked_fill(mask, -self.neg_large)

        p_t = F.softmax(sim_t / self.tau_t, dim=1).clamp_min(self.eps)
        log_p_s = F.log_softmax(sim_s / self.tau_s, dim=1)

        # input=log_probs, target=probs
        return F.kl_div(log_p_s, p_t, reduction=self.reduction)


class NTXentLoss(nn.Module):
    """
    SimCLR-style NT-Xent, stable & backend-friendly (CPU/CUDA/MPS).
    Assumes z1,z2 are [B,D]. If not normalized, we normalize here for safety.
    """
    def __init__(self, temperature: float = 0.2, neg_large: float = 1e9):
        super().__init__()
        self.temperature = float(temperature)
        self.neg_large = float(neg_large)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        assert z1.dim() == 2 and z2.dim() == 2 and z1.shape == z2.shape
        B, _ = z1.shape
        if B < 2:
            return z1.new_tensor(0.0)

        # normalize to cosine sim
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        z = torch.cat([z1, z2], dim=0)  # [2B, D]
        logits = (z @ z.t()) / self.temperature  # [2B, 2B]

        # mask self-contrast (diagonal) without in-place, avoid -inf
        mask = torch.eye(2 * B, device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(mask, -self.neg_large)

        # positive index mapping:
        # for i in [0..B-1], pos is i+B
        # for i in [B..2B-1], pos is i-B
        pos_index = torch.arange(2 * B, device=logits.device)
        pos_index = (pos_index + B) % (2 * B)

        # cross-entropy over rows (each row picks its positive column)
        loss = F.cross_entropy(logits, pos_index)
        return loss
