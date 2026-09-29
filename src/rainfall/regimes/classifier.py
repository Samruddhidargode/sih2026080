"""
Stage 2b - Regime classifier: the CNN "gatekeeper".

Input : synoptic channels only (u850, v850, MSLP), z-scored, 3 x 129 x 135.
        Never rainfall - so the same network runs on tomorrow's forecast.
Arch  : [Conv3x3 -> BatchNorm -> ReLU -> MaxPool] x 4 -> global avg pool -> dense -> softmax(4)
        ~60k params. Deliberately small: ~250 training days, and the rare
        Depression class has only a few dozen. A big network would memorise them.
Loss  : class-weighted cross-entropy (weights = inverse class frequency) so
        the rare classes are not drowned by Coastal/Orographic days.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rainfall.regimes.labels import CLASSES


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(2))


class RegimeCNN(nn.Module):
    def __init__(self, in_ch: int = 3, n_classes: int = len(CLASSES), width: int = 16, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(_block(in_ch, width), _block(width, width * 2),
                                      _block(width * 2, width * 4), _block(width * 4, width * 8))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(width * 8, 64), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                  nn.Linear(64, n_classes))

    def forward(self, x):            # returns logits; softmax applied by the caller
        return self.head(self.features(x))


def class_weights(labels: np.ndarray, n_classes: int = len(CLASSES)) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype("float64")
    w = counts.sum() / np.maximum(counts, 1) / n_classes
    return torch.tensor(w, dtype=torch.float32)


def train_classifier(Xtr, ytr, Xva, yva, epochs=80, lr=1e-3, batch=16, device="cpu", seed=42, weight_decay=1e-4):
    """Xtr: (N,3,H,W) float32 normalised; ytr: (N,) int. Returns (model, history)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = RegimeCNN(in_ch=Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    wts = class_weights(ytr).to(device)
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.long)
    Xva_t, yva_t = torch.tensor(Xva).to(device), torch.tensor(yva, dtype=torch.long).to(device)
    best, best_state, hist = -1.0, None, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        tot = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr_t[idx].to(device), ytr_t[idx].to(device)
            # light augmentation: random horizontal jitter of a few cells - synoptic patterns are not pixel-locked
            if np.random.rand() < 0.5:
                xb = torch.roll(xb, shifts=int(np.random.randint(-3, 4)), dims=3)
            loss = F.cross_entropy(model(xb), yb, weight=wts)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva_t).argmax(1)
            acc = float((pv == yva_t).float().mean())
            # balanced accuracy = mean per-class recall - the metric that matters with imbalance
            recalls = [float((pv[yva_t == c] == c).float().mean()) for c in range(len(CLASSES)) if (yva_t == c).any()]
            bal = float(np.mean(recalls))
        hist.append(dict(epoch=ep, train_loss=tot / len(perm), val_acc=acc, val_bal_acc=bal))
        if bal > best:
            best, best_state = bal, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, hist


@torch.no_grad()
def predict_proba(model, X, device="cpu", batch=64) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        out.append(F.softmax(model(torch.tensor(X[i:i + batch]).to(device)), 1).cpu().numpy())
    return np.concatenate(out)


def evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Confusion matrix + per-class precision/recall/f1, no sklearn dependency
    (sklearn's OpenMP DLL is blocked by some Windows Application Control policies)."""
    pred = proba.argmax(1)
    n = len(CLASSES)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, pred):
        cm[t, p] += 1
    report = {}
    for k, name in enumerate(CLASSES):
        tp = cm[k, k]; fp = cm[:, k].sum() - tp; fn = cm[k, :].sum() - tp; support = int(cm[k, :].sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        report[name] = {"precision": prec, "recall": rec, "f1-score": f1, "support": support}
    return {"report": report, "confusion": cm.tolist(), "accuracy": float((pred == y_true).mean())}
