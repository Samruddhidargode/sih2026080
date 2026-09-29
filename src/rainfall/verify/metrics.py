"""
Stage 5 - Verification: raw NWP vs corrected vs IMD.

RMSE, POD, FAR, CSI, ETS via xskillscore (2x2 contingency at each IMD threshold);
FSS (Roberts & Lean 2008) via a neighbourhood fraction implementation
(xskillscore has no FSS). Everything is computed only where IMD has data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import xskillscore as xs
from scipy.ndimage import uniform_filter

from rainfall.grid import LAT, LON

THRESHOLDS = {"heavy (>64.5 mm)": 64.5, "very heavy (>115.5 mm)": 115.5, "extremely heavy (>204.4 mm)": 204.4}
FSS_SCALES = (1, 3, 5, 9)     # cells; 0.25 deg ~ 27 km -> 27, 80, 135, 245 km


def _da(a: np.ndarray, name: str) -> xr.DataArray:
    return xr.DataArray(a, dims=("sample", "lat", "lon"), coords={"lat": LAT, "lon": LON}, name=name)


def contingency_scores(fcst: np.ndarray, obs: np.ndarray, thr: float) -> dict:
    """POD, FAR, CSI, ETS over all valid cells and samples, pooled."""
    mask = np.isfinite(obs)
    f = _da(np.where(mask, fcst, np.nan), "f"); o = _da(np.where(mask, obs, np.nan), "o")
    edges = np.array([-np.inf, thr, np.inf])
    c = xs.Contingency(o, f, edges, edges, dim=("sample", "lat", "lon"))
    return {"POD": float(c.hit_rate()), "FAR": float(c.false_alarm_ratio()),
            "CSI": float(c.threat_score()), "ETS": float(c.equit_threat_score()),
            "hits": int(c.hits()), "misses": int(c.misses()), "false_alarms": int(c.false_alarms())}


def rmse(fcst: np.ndarray, obs: np.ndarray) -> float:
    mask = np.isfinite(obs)
    return float(np.sqrt(np.mean((fcst[mask] - obs[mask]) ** 2)))


def fss(fcst: np.ndarray, obs: np.ndarray, thr: float, scale: int) -> float:
    """Fractions Skill Score for a (sample, lat, lon) stack. Cells with no obs are
    treated as 'no event' for the neighbourhood fraction (they are ocean/outside India)."""
    mask = np.isfinite(obs)
    fo = (np.nan_to_num(obs) > thr).astype("float32"); ff = (fcst > thr).astype("float32")
    num = den = 0.0
    for i in range(len(obs)):
        po = uniform_filter(fo[i], size=scale, mode="constant") if scale > 1 else fo[i]
        pf = uniform_filter(ff[i], size=scale, mode="constant") if scale > 1 else ff[i]
        m = mask[i]
        num += np.sum((pf[m] - po[m]) ** 2); den += np.sum(pf[m] ** 2 + po[m] ** 2)
    return float(1 - num / den) if den > 0 else np.nan


def score_table(forecasts: dict[str, np.ndarray], obs: np.ndarray, scales=FSS_SCALES) -> pd.DataFrame:
    """forecasts: {"Raw NWP": (N,H,W), "Corrected": (N,H,W)} -> long table metric x threshold x forecast."""
    rows = []
    for name, f in forecasts.items():
        rows.append(dict(forecast=name, threshold="all", metric="RMSE (mm)", value=rmse(f, obs)))
        for tname, thr in THRESHOLDS.items():
            c = contingency_scores(f, obs, thr)
            for k in ("POD", "FAR", "CSI", "ETS"):
                rows.append(dict(forecast=name, threshold=tname, metric=k, value=c[k]))
            for s in scales:
                rows.append(dict(forecast=name, threshold=tname, metric=f"FSS {s}x{s}", value=fss(f, obs, thr, s)))
    return pd.DataFrame(rows)


def side_by_side(table: pd.DataFrame, raw="Raw NWP", corr="Corrected") -> pd.DataFrame:
    w = table.pivot_table(index=["threshold", "metric"], columns="forecast", values="value").reset_index()
    if raw in w and corr in w:
        better_low = w["metric"].isin(["RMSE (mm)", "FAR"])
        w["improvement"] = np.where(better_low, w[raw] - w[corr], w[corr] - w[raw])
    return w


def brier(prob: np.ndarray, obs: np.ndarray, thr: float) -> float:
    mask = np.isfinite(obs)
    return float(np.mean((prob[mask] - (obs[mask] > thr)) ** 2))


def reliability(prob: np.ndarray, obs: np.ndarray, thr: float, bins=10) -> pd.DataFrame:
    mask = np.isfinite(obs)
    p, o = prob[mask], (obs[mask] > thr).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = [dict(bin_centre=(edges[b] + edges[b + 1]) / 2, forecast_prob=p[idx == b].mean() if (idx == b).any() else np.nan,
                 observed_freq=o[idx == b].mean() if (idx == b).any() else np.nan, count=int((idx == b).sum())) for b in range(bins)]
    return pd.DataFrame(rows)
