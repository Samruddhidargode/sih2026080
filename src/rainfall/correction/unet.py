"""
Stage 3 - Correction U-Net + asymmetric multi-task loss + regime-routed experts.

One U-Net class serves all four experts; the per-regime differences from the
design notes are constructor arguments:
    Active      : kernel 3, standard
    Break       : kernel 3, trained with L2 weight decay (sparse / zero-rain targets)
    Depression  : kernel 7 in the encoder (wider receptive field for synoptic rain bands)
    Coastal/Oro : dense skips (skip connections at every level are standard U-Net;
                  we additionally keep the full-resolution input as an extra skip)

Output head (identical on every expert): 4 channels
    ch0    log1p(rain) via softplus  -> rain_mm = expm1(ch0)   (never negative)
    ch1-3  logits of P(rain > 64.5 / 115.5 / 204.4 mm)         (sigmoid at inference)

Loss (masked where IMD is NaN):
    L = MSE(log1p y, ch0)
      + lam * 1[y > 64.5] * relu(log1p y - ch0)^2      <- extra cost ONLY for under-predicting heavy cells
      + sum_k BCE_with_logits(ch_k, 1[y > t_k], pos_weight)
Working in log1p space keeps a 300 mm cell from dominating the gradient while
the asymmetric term still says: missing a heavy-rain cell is the expensive mistake.
"""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

THRESHOLDS = (64.5, 115.5, 204.4)


def _conv(cin, cout, k=3):
    p = k // 2
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=p), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, in_ch=6, out_ch=4, base=32, enc_kernel=3, input_skip=False):
        super().__init__()
        b = base
        self.input_skip = input_skip
        self.e1 = _conv(in_ch, b, enc_kernel); self.e2 = _conv(b, 2 * b, enc_kernel)
        self.e3 = _conv(2 * b, 4 * b, enc_kernel); self.e4 = _conv(4 * b, 8 * b, 3)
        self.pool = nn.MaxPool2d(2)
        self.u3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, stride=2); self.d3 = _conv(8 * b, 4 * b)
        self.u2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, stride=2); self.d2 = _conv(4 * b, 2 * b)
        self.u1 = nn.ConvTranspose2d(2 * b, b, 2, stride=2); self.d1 = _conv(2 * b + (in_ch if input_skip else 0), b)
        self.out = nn.Conv2d(b, out_ch, 1)

    @staticmethod
    def _match(x, ref):
        """129x135 is not divisible by 8 - crop/pad the upsampled map to the skip's size."""
        dy, dx = ref.shape[-2] - x.shape[-2], ref.shape[-1] - x.shape[-1]
        return F.pad(x, (0, dx, 0, dy)) if (dy >= 0 and dx >= 0) else x[..., :ref.shape[-2], :ref.shape[-1]]

    def forward(self, x):
        s1 = self.e1(x); s2 = self.e2(self.pool(s1)); s3 = self.e3(self.pool(s2)); bt = self.e4(self.pool(s3))
        d = self.d3(torch.cat([self._match(self.u3(bt), s3), s3], 1))
        d = self.d2(torch.cat([self._match(self.u2(d), s2), s2], 1))
        d = self._match(self.u1(d), s1)
        d = self.d1(torch.cat([d, s1, x], 1) if self.input_skip else torch.cat([d, s1], 1))
        z = self.out(d)
        return torch.cat([F.softplus(z[:, :1]), z[:, 1:]], 1)   # ch0 = log1p(rain) >= 0, ch1-3 logits


EXPERT_KWARGS = {
    "Active":             dict(enc_kernel=3),
    "Break":              dict(enc_kernel=3),          # + weight_decay in the optimiser
    "Depression":         dict(enc_kernel=7),
    "Coastal/Orographic": dict(enc_kernel=3, input_skip=True),
}
EXPERT_WEIGHT_DECAY = {"Active": 1e-5, "Break": 1e-3, "Depression": 1e-5, "Coastal/Orographic": 1e-5}


# ---------- loss -------------------------------------------------------------

def asymmetric_loss(pred, y, lam=3.0, heavy=64.5, pos_weights=(20., 50., 100.)):
    """pred: (B,4,H,W) network output; y: (B,H,W) rain mm with NaN where no obs."""
    mask = torch.isfinite(y)
    y0 = torch.where(mask, y, torch.zeros_like(y))
    ly = torch.log1p(y0.clamp(min=0))
    p0 = pred[:, 0]
    m = mask.float()
    n = m.sum().clamp(min=1)
    mse = (((p0 - ly) ** 2) * m).sum() / n
    heavy_mask = (y0 > heavy).float() * m
    miss = (F.relu(ly - p0) ** 2 * heavy_mask).sum() / heavy_mask.sum().clamp(min=1)
    bce = 0.0
    for k, (t, pw) in enumerate(zip(THRESHOLDS, pos_weights)):
        target = (y0 > t).float()
        l = F.binary_cross_entropy_with_logits(pred[:, 1 + k], target, pos_weight=torch.tensor(pw, device=pred.device), reduction="none")
        bce = bce + (l * m).sum() / n
    return mse + lam * miss + bce, dict(mse=float(mse.detach()), miss=float(miss.detach()), bce=float(bce.detach()) if torch.is_tensor(bce) else float(bce))


# ---------- training ----------------------------------------------------------

def train_unet(model, Xtr, ytr, Xva, yva, epochs=40, lr=1e-3, batch=8, weight_decay=1e-5, lam=3.0,
               device="cpu", seed=42, patience=10, log=print):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    Xva_t, yva_t = torch.tensor(Xva).to(device), torch.tensor(yva).to(device)
    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xtr_t)); tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr_t[idx].to(device), ytr_t[idx].to(device)
            loss, _ = asymmetric_loss(model(xb), yb, lam=lam)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += float(loss) * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = 0.0
            for i in range(0, len(Xva_t), 16):
                l, _ = asymmetric_loss(model(Xva_t[i:i + 16]), yva_t[i:i + 16], lam=lam); vl += float(l) * len(Xva_t[i:i + 16])
            vl /= max(len(Xva_t), 1)
        hist.append(dict(epoch=ep, train=tot / len(perm), val=vl))
        log(f"  ep {ep:3d} train {tot / len(perm):.4f}  val {vl:.4f}")
        if vl < best - 1e-4:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, hist


@torch.no_grad()
def predict(model, X, device="cpu", batch=16):
    """Returns rain_mm (N,H,W) and probs (N,3,H,W)."""
    model.eval(); rain, prob = [], []
    for i in range(0, len(X), batch):
        out = model(torch.tensor(X[i:i + batch]).to(device))
        rain.append(torch.expm1(out[:, 0]).cpu().numpy()); prob.append(torch.sigmoid(out[:, 1:]).cpu().numpy())
    rain = np.concatenate(rain); prob = np.concatenate(prob)
    prob = np.minimum.accumulate(prob, axis=1)     # enforce P(>64.5) >= P(>115.5) >= P(>204.4)
    return rain, prob


def make_expert(regime: str, base=32, in_ch=6) -> UNet:
    return UNet(in_ch=in_ch, base=base, **EXPERT_KWARGS[regime])
