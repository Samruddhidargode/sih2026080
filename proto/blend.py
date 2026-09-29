"""
Stage 3: soft, regime-weighted blending of the four experts + probability calibration.

Weights are learned per situation (12 = day regime x local type) on OUT-OF-FOLD
expert predictions only (each dev year predicted by models that never saw it),
shrunk towards the all-India weights when a situation has little data, and then
mixed per box using the situation PROBABILITIES - no hard switching.
"""
from __future__ import annotations

import itertools

import numpy as np

from common import THR

COMBOS = np.array([c for c in itertools.product(range(11), repeat=4) if sum(c) == 10], float) / 10.0  # 286


def _best(A, target, kind):
    """A: (4, m) expert values on m rows; returns the simplex weights minimising the loss."""
    best, bw = np.inf, None
    for s in range(0, len(COMBOS), 32):
        C = COMBOS[s:s + 32]
        pred = C @ A                                             # (k, m)
        loss = ((pred - target) ** 2).mean(1) if kind == "amt" else ((pred - target) ** 2).mean(1)
        i = int(np.argmin(loss))
        if loss[i] < best:
            best, bw = float(loss[i]), C[i]
    return bw


class SoftBlender:
    def __init__(self, n0: float = 20000.0):
        self.n0 = n0

    def fit(self, A, P, S, obs, max_rows=400_000, seed=0):
        """A (4,n,H,W) amounts, P (4,3,n,H,W) probs, S (12,n,H,W) situation probs, obs (n,H,W)."""
        rng = np.random.default_rng(seed)
        ok = np.isfinite(obs) & np.isfinite(A).all(0) & np.isfinite(P).all((0, 1))
        hard = np.argmax(np.nan_to_num(S), 0)
        a = A[:, ok]; p = P[:, 0, ok]; p2 = P[:, 1, ok]; o = obs[ok]; h = hard[ok]
        yb, yb2 = (o > THR[0]).astype(float), (o > THR[1]).astype(float)

        def fit_rows(sel):
            if sel.sum() > max_rows:
                sel = np.where(sel)[0]; sel = rng.choice(sel, max_rows, replace=False)
            wa = _best(a[:, sel], o[sel], "amt")
            # probability weights: Brier at heavy + very heavy (stack both thresholds)
            wp = _best(np.concatenate([p[:, sel], p2[:, sel]], 1), np.concatenate([yb[sel], yb2[sel]]), "prob")
            return wa, wp

        self.global_w = fit_rows(np.ones(len(o), bool))
        self.w_amt, self.w_prob, self.n = [], [], []
        for c in range(S.shape[0]):
            sel = h == c; m = int(sel.sum()); self.n.append(m)
            if m < 200:
                wa, wp = self.global_w
            else:
                wa, wp = fit_rows(sel)
            lam = m / (m + self.n0)
            self.w_amt.append(lam * wa + (1 - lam) * self.global_w[0])
            self.w_prob.append(lam * wp + (1 - lam) * self.global_w[1])
        self.w_amt, self.w_prob = np.array(self.w_amt), np.array(self.w_prob)   # (12, 4)
        return self

    def box_weights(self, S):
        """(4, n, H, W) per-box expert weights = sum_s P(s) * w_s."""
        return np.einsum("snhw,se->enhw", np.nan_to_num(S), self.w_amt), \
            np.einsum("snhw,se->enhw", np.nan_to_num(S), self.w_prob)

    def apply(self, A, P, S):
        Wa, Wp = self.box_weights(S)
        amt = (Wa * A).sum(0)
        prob = (Wp[:, None] * P).sum(0)
        return amt.astype("float32"), np.minimum.accumulate(prob, axis=0).astype("float32"), Wa


# ----------------------------------------------------------------------------- calibration
def _pav(x, y, w):
    """Pool-adjacent-violators isotonic regression (x sorted)."""
    v, wt, cnt = list(y.astype(float)), list(w.astype(float)), [1] * len(y)
    i = 0
    while i < len(v) - 1:
        if v[i] > v[i + 1] + 1e-12:
            tw = wt[i] + wt[i + 1]; v[i] = (v[i] * wt[i] + v[i + 1] * wt[i + 1]) / tw; wt[i] = tw
            cnt[i] += cnt[i + 1]; del v[i + 1], wt[i + 1], cnt[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return np.repeat(v, cnt)


class Calibrator:
    """Isotonic (heavy, very heavy) / logistic (extremely heavy, few events) per threshold."""

    def fit(self, P, obs, nbins=200):
        ok = np.isfinite(obs) & np.isfinite(P).all(0)
        self.maps = []
        for k, t in enumerate(THR):
            x = P[k][ok]; yb = (obs[ok] > t).astype(float)
            if k < 2:
                edges = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
                b = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
                w = np.bincount(b, minlength=len(edges) - 1)
                keep = w > 0
                xm = np.bincount(b, x, len(edges) - 1)[keep] / w[keep]
                ym = np.bincount(b, yb, len(edges) - 1)[keep] / w[keep]
                self.maps.append(("iso", xm, _pav(xm, ym, w[keep])))
            else:
                z = np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
                a, b = 1.0, 0.0
                for _ in range(50):                                # Newton for 2-parameter Platt scaling
                    p = 1 / (1 + np.exp(-(a * z + b))); g = p - yb; h = p * (1 - p) + 1e-9
                    H = np.array([[np.sum(h * z * z), np.sum(h * z)], [np.sum(h * z), np.sum(h)]]) + 1e-6 * np.eye(2)
                    step = np.linalg.solve(H, np.array([np.sum(g * z), np.sum(g)]))
                    a, b = a - step[0], b - step[1]
                self.maps.append(("logit", a, b))
        return self

    def apply(self, P):
        out = np.full_like(P, np.nan)
        for k, m in enumerate(self.maps):
            x = P[k]
            if m[0] == "iso":
                out[k] = np.interp(np.nan_to_num(x), m[1], m[2])
            else:
                z = np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
                out[k] = 1 / (1 + np.exp(-(m[1] * z + m[2])))
            out[k][~np.isfinite(x)] = np.nan
        return np.minimum.accumulate(out, axis=0)
