"""
The four correction experts. Every expert returns the same format:
    amount (n, H, W)       corrected 24-h rain, mm
    probs  (3, n, H, W)    P(rain > 64.5 / 115.5 / 204.4 mm)

E0  Box quantile mapping    - one correction for every day  (the "single method" baseline)
E1  Regime-aware QM         - E0's box climatology + a rank adjustment learned per situation
E2  LightGBM                - physics features -> expected amount + exceedance probabilities
E3  U-Net (CNN)             - whole-map correction; can move misplaced rain (existing code)
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import torch

from common import THR

BINS = np.array([0, 0.5, 2.5, 5, 10, 15.6, 25, 35, 50, 64.5, 80, 100, 115.5, 150, 204.4, 300, 1e9])


# ----------------------------------------------------------------------------- E0
def _neighbour_stack(a: np.ndarray, r: int) -> np.ndarray:
    n, H, W = a.shape
    pad = np.pad(a, ((0, 0), (r, r), (r, r)), constant_values=np.nan)
    return np.concatenate([pad[:, r + dy:r + dy + H, r + dx:r + dx + W]
                           for dy in range(-r, r + 1) for dx in range(-r, r + 1)], 0)


class BoxQM:
    """Per-box empirical CDFs of forecast and observed rain (5x5-box neighbourhood pooled)."""

    def __init__(self, r: int = 2):
        self.r = r

    def fit(self, fc, obs, domain):
        self.shape = domain.shape; self.idx = np.where(domain.ravel())[0]
        F = np.where(np.isfinite(obs), fc, np.nan)
        iy, ix = np.unravel_index(self.idx, domain.shape); r = self.r; H, W = domain.shape
        Fp = np.pad(F, ((0, 0), (r, r), (r, r)), constant_values=np.nan)
        Op = np.pad(obs, ((0, 0), (r, r), (r, r)), constant_values=np.nan)
        offs = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
        Fs = np.concatenate([Fp[:, iy + r + dy, ix + r + dx] for dy, dx in offs], 0)   # only IMD boxes
        Os = np.concatenate([Op[:, iy + r + dy, ix + r + dx] for dy, dx in offs], 0)
        Fs.sort(0); Os.sort(0)                                     # NaN sorted to the end
        self.nF = np.isfinite(Fs).sum(0); self.nO = np.isfinite(Os).sum(0)
        self.F, self.O = Fs.astype("float32"), Os.astype("float32")
        self.Fmax = np.array([self.F[k - 1, j] for j, k in enumerate(self.nF)])
        self.Omax = np.array([self.O[k - 1, j] for j, k in enumerate(self.nO)])
        return self

    def _flat(self, a):
        return a.reshape(len(a), -1)[:, self.idx]

    def _grid(self, b):
        g = np.full((len(b), int(np.prod(self.shape))), np.nan, "float32")
        g[:, self.idx] = b
        return g.reshape((len(b),) + self.shape)

    def rank(self, v, which="F"):
        """Mid-rank empirical CDF value in [0,1] (ties at zero handled fairly)."""
        S, N = (self.F, self.nF) if which == "F" else (self.O, self.nO)
        vb = self._flat(v); u = np.full(vb.shape, np.nan, "float32")
        for j in range(vb.shape[1]):
            col = S[:N[j], j]
            ok = np.isfinite(vb[:, j])
            x = vb[ok, j]
            u[ok, j] = (np.searchsorted(col, x, "left") + np.searchsorted(col, x, "right")) / (2.0 * N[j])
        return self._grid(u)

    def inv_obs(self, u, fc=None):
        """Observed-rain quantile at CDF value u; beyond the largest forecast ever seen, scale up."""
        ub = self._flat(u); out = np.full(ub.shape, np.nan, "float32")
        fb = self._flat(fc) if fc is not None else None
        for j in range(ub.shape[1]):
            col = self.O[:self.nO[j], j]
            ok = np.isfinite(ub[:, j])
            pos = np.clip(ub[ok, j], 0, 1) * (self.nO[j] - 1)
            out[ok, j] = np.interp(pos, np.arange(self.nO[j]), col)
            if fb is not None and self.Fmax[j] > 0:             # extrapolate record-breaking forecasts
                ext = ok & (fb[:, j] > self.Fmax[j])
                out[ext, j] = self.Omax[j] * np.clip(fb[ext, j] / self.Fmax[j], 1, 2)
        return self._grid(out)

    def map(self, fc):
        return self.inv_obs(self.rank(fc, "F"), fc)


def _exceed_table(val, obs, prior=None, a=50.0):
    """P(obs > t | bin of corrected value) with a Bayesian prior (shrinkage for sparse bins)."""
    ok = np.isfinite(val) & np.isfinite(obs)
    b = np.clip(np.digitize(val[ok], BINS) - 1, 0, len(BINS) - 2); o = obs[ok]
    n = np.bincount(b, minlength=len(BINS) - 1).astype(float)
    tab = np.zeros((len(BINS) - 1, len(THR)))
    for k, t in enumerate(THR):
        hit = np.bincount(b, weights=(o > t).astype(float), minlength=len(BINS) - 1)
        base = prior[:, k] if prior is not None else np.full(len(n), hit.sum() / max(n.sum(), 1))
        tab[:, k] = (hit + a * base) / (n + a)
    return np.minimum.accumulate(tab, axis=1), n


def _lookup(tab, val):
    b = np.clip(np.digitize(np.nan_to_num(val), BINS) - 1, 0, len(BINS) - 2)
    p = np.moveaxis(tab[b], -1, 0).astype("float32")
    p[:, ~np.isfinite(val)] = np.nan
    return p


class E0QuantileMap:
    name = "E0"

    def fit(self, fc, obs, domain):
        self.qm = BoxQM().fit(fc, obs, domain)
        self.tab, _ = _exceed_table(self.qm.map(fc), obs)
        return self

    def predict(self, fc):
        amt = self.qm.map(fc)
        return amt, _lookup(self.tab, amt)


# ----------------------------------------------------------------------------- E1
LV = np.linspace(0, 1, 201)


class E1RegimeQM:
    """Keeps each box's own climatology (from E0) but learns, for each of the 12
    situations, how the forecast's rank should be shifted - e.g. 'on coastal boxes
    in an active spell the model's 90th percentile is really the observed 97th'."""
    name = "E1"

    def __init__(self, qm: BoxQM, n0: float = 3000.0):
        self.qm, self.n0 = qm, n0

    def fit(self, fc, obs, S):
        uf = self.qm.rank(fc, "F"); uo = self.qm.rank(obs, "O")
        hard = np.argmax(np.nan_to_num(S), 0)
        ok = np.isfinite(uf) & np.isfinite(uo)
        glob_tab, _ = _exceed_table(self.qm.inv_obs(uf, fc), obs)
        self.maps, self.tabs, self.n = [], [], []
        for c in range(S.shape[0]):
            sel = ok & (hard == c); m = int(sel.sum())
            if m > 50:
                qf = np.maximum.accumulate(np.quantile(uf[sel], LV)) + LV * 1e-6
                qo = np.maximum.accumulate(np.quantile(uo[sel], LV))
            else:
                qf, qo = LV.copy(), LV.copy()
            w = m / (m + self.n0)
            self.maps.append((qf, qo, w)); self.n.append(m)
            val = self._apply(uf, fc, c)
            tab, _ = _exceed_table(np.where(hard == c, val, np.nan), obs, prior=glob_tab)
            self.tabs.append(tab)
        return self

    def _apply(self, uf, fc, c):
        qf, qo, w = self.maps[c]
        g = w * np.interp(np.nan_to_num(uf), qf, qo) + (1 - w) * np.nan_to_num(uf)
        g = np.where(np.isfinite(uf), g, np.nan)
        return self.qm.inv_obs(g, fc)

    def predict(self, fc, S):
        uf = self.qm.rank(fc, "F")
        amt = np.zeros(fc.shape, "float32"); prob = np.zeros((len(THR),) + fc.shape, "float32")
        for c in range(S.shape[0]):
            val = self._apply(uf, fc, c)
            amt += S[c] * np.nan_to_num(val); prob += S[c][None] * np.nan_to_num(_lookup(self.tabs[c], val))
        mask = ~np.isfinite(uf)
        amt[mask] = np.nan; prob[:, mask] = np.nan
        return amt, np.minimum.accumulate(prob, axis=0)


# ----------------------------------------------------------------------------- E2
GBM = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=300, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7, num_threads=0)


class E2LightGBM:
    name = "E2"

    def fit(self, feats: dict, obs_flat: np.ndarray, rounds=300):
        self.cols = list(feats)
        Xf = np.stack([feats[c].ravel() for c in self.cols], 1)
        yv = obs_flat.ravel(); ok = np.isfinite(yv); Xf, yv = Xf[ok], yv[ok]
        self.m_amt = lgb.train({**GBM, "objective": "tweedie", "tweedie_variance_power": 1.3},
                               lgb.Dataset(Xf, yv), rounds)
        self.m_p = [lgb.train({**GBM, "objective": "binary"}, lgb.Dataset(Xf, (yv > t).astype(int)), rounds)
                    for t in THR]
        return self

    def predict(self, feats: dict, domain):
        n = feats[self.cols[0]].shape[0]
        Xf = np.stack([feats[c].ravel() for c in self.cols], 1)
        idx = np.where(domain.ravel())[0]

        def grid(v):
            g = np.full((n, domain.size), np.nan, "float32"); g[:, idx] = v.reshape(n, -1)
            return g.reshape((n,) + domain.shape)
        amt = grid(self.m_amt.predict(Xf))
        prob = np.stack([grid(m.predict(Xf)) for m in self.m_p])
        return amt, np.minimum.accumulate(prob, axis=0)

    def importance(self):
        g = self.m_p[0].feature_importance("gain")
        return dict(sorted(zip(self.cols, (g / g.sum()).round(4)), key=lambda kv: -kv[1]))


# ----------------------------------------------------------------------------- E3
class E3UNet:
    """Wraps the project's existing U-Net (rainfall.correction.unet); retrained per fold."""
    name = "E3"

    def __init__(self, epochs=40, base=32):
        self.epochs, self.base = epochs, base
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, X, y):
        from rainfall.correction.unet import UNet, train_unet
        from rainfall.data.build_dataset import ChannelNormalizer
        self.norm = ChannelNormalizer().fit(X)
        Xn = self.norm.transform(X)
        nv = max(int(0.15 * len(X)), 8)                           # last 15% of days: early stopping
        self.model, _ = train_unet(UNet(in_ch=X.shape[1], base=self.base), Xn[:-nv], y[:-nv], Xn[-nv:], y[-nv:],
                                   epochs=self.epochs, device=self.dev, patience=8, log=lambda *a: None)
        return self

    def predict(self, X, domain):
        from rainfall.correction.unet import predict
        rain, prob = predict(self.model, self.norm.transform(X), self.dev)
        rain[:, ~domain] = np.nan; prob[:, :, ~domain] = np.nan
        return rain.astype("float32"), np.moveaxis(prob, 1, 0).astype("float32")
