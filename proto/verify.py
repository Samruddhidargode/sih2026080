"""Verification scores (NumPy only): RMSE, POD, FAR, CSI, ETS, frequency bias, FSS, Brier/BSS."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def counts(yes_f, yes_o, mask):
    f, o = yes_f & mask, yes_o & mask
    return np.array([(f & o).sum(), (~f & o).sum(), (f & ~o).sum(), (~f & ~o & mask).sum()], float)


def cat_scores(c):
    h, m, fa, cn = c; n = c.sum()
    hr = (h + fa) * (h + m) / max(n, 1)
    return dict(POD=h / max(h + m, 1), FAR=fa / max(h + fa, 1), CSI=h / max(h + m + fa, 1),
                ETS=(h - hr) / max(h + m + fa - hr, 1e-9), FreqBias=(h + fa) / max(h + m, 1))


def fss(yes_f, yes_o, mask, size):
    num, den = 0.0, 0.0
    for i in range(len(yes_f)):
        ff = uniform_filter((yes_f[i] & mask[i]).astype(float), size)
        fo = uniform_filter((yes_o[i] & mask[i]).astype(float), size)
        mm = mask[i]
        num += ((ff - fo) ** 2)[mm].sum(); den += (ff ** 2 + fo ** 2)[mm].sum()
    return 1 - num / den if den > 0 else np.nan


def brier(p, yes_o, mask):
    return float(np.mean((p[mask] - yes_o[mask]) ** 2))


def season_table(methods: dict, obs, clim_freq: dict, mask=None, thr_list=(64.5, 115.5)):
    """methods: name -> dict(amount, yes={thr: bool array}, prob={thr: array or None})."""
    base = np.isfinite(obs) if mask is None else (np.isfinite(obs) & mask)
    rows = []
    for name, m in methods.items():
        ok = base & np.isfinite(m["amount"])
        r = dict(method=name, RMSE=float(np.sqrt(np.mean((m["amount"][ok] - obs[ok]) ** 2))),
                 Bias=float(m["amount"][ok].sum() / obs[ok].sum()))
        for t in thr_list:
            yo = obs > t
            s = cat_scores(counts(m["yes"][t], yo, ok))
            for k, v in s.items():
                r[f"{k}@{t}"] = float(v)
            if t == thr_list[0]:
                r[f"FSS3@{t}"] = float(fss(m["yes"][t], yo, ok, 3))
                r[f"FSS9@{t}"] = float(fss(m["yes"][t], yo, ok, 9))
            p = m["prob"].get(t) if m.get("prob") else None
            if p is not None:
                bs = brier(p, yo.astype(float), ok)
                bc = brier(np.broadcast_to(clim_freq[t], obs.shape), yo.astype(float), ok)
                r[f"Brier@{t}"] = bs; r[f"BSS@{t}"] = 1 - bs / bc
        rows.append(r)
    return rows


def best_threshold(p, obs, t, grid=np.arange(0.05, 0.91, 0.025)):
    """Probability cut-off that maximises CSI (tuned on out-of-fold data only)."""
    ok = np.isfinite(p) & np.isfinite(obs); yo = obs > t
    scores = [cat_scores(counts(p >= g, yo, ok))["CSI"] for g in grid]
    return float(grid[int(np.argmax(scores))])


def block_bootstrap_csi(yes_a, yes_b, obs, t, mask, block=5, n=1000, seed=0):
    """95% CI of CSI(a) - CSI(b), resampling 5-day blocks (monsoon spells are autocorrelated)."""
    ca = np.array([counts(yes_a[i:i + 1], obs[i:i + 1] > t, mask[i:i + 1]) for i in range(len(obs))])
    cb = np.array([counts(yes_b[i:i + 1], obs[i:i + 1] > t, mask[i:i + 1]) for i in range(len(obs))])
    rng = np.random.default_rng(seed); nb = int(np.ceil(len(obs) / block)); diffs = []
    starts = np.arange(0, len(obs) - block + 1)
    for _ in range(n):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, nb)])[:len(obs)]
        diffs.append(cat_scores(ca[idx].sum(0))["CSI"] - cat_scores(cb[idx].sum(0))["CSI"])
    d = cat_scores(ca.sum(0))["CSI"] - cat_scores(cb.sum(0))["CSI"]
    return dict(diff=float(d), lo=float(np.percentile(diffs, 2.5)), hi=float(np.percentile(diffs, 97.5)))
