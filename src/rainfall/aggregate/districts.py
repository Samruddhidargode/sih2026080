"""
Stage 4 - District aggregation.

A (districts x cells) sparse weight matrix is built ONCE from the shapefile:
w[d, c] = area of cell c inside district d / area of district d.
Aggregating any grid is then one matrix product (mean) or a masked max (flood flag).
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
from shapely.geometry import box

from rainfall.grid import LAT, LON, NLAT, NLON

log = logging.getLogger(__name__)


def cell_polygons() -> gpd.GeoDataFrame:
    cells = [box(lo - 0.125, la - 0.125, lo + 0.125, la + 0.125) for la in LAT for lo in LON]
    return gpd.GeoDataFrame({"cell": np.arange(NLAT * NLON)}, geometry=cells, crs="EPSG:4326")


def build_weights(districts: gpd.GeoDataFrame, cache: str | Path | None = None) -> sp.csr_matrix:
    if cache and Path(cache).exists():
        return sp.load_npz(cache)
    log.info("building district x cell weights (one-time)")
    cells = cell_polygons()
    # equal-area projection so 'area' means km^2, not degrees^2
    d = districts[["district_id", "geometry"]].to_crs("EPSG:7755")
    c = cells.to_crs("EPSG:7755")
    inter = gpd.overlay(d, c, how="intersection", keep_geom_type=False)
    inter["a"] = inter.geometry.area
    tot = inter.groupby("district_id")["a"].transform("sum")
    w = sp.csr_matrix((inter["a"] / tot, (inter["district_id"], inter["cell"])), shape=(len(districts), NLAT * NLON))
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True); sp.save_npz(cache, w)
    return w


def aggregate(grid: np.ndarray, W: sp.csr_matrix, how: str = "mean") -> np.ndarray:
    """grid (lat, lon) -> per-district values. NaN cells are ignored (renormalised)."""
    v = grid.reshape(-1).astype("float64")
    valid = np.isfinite(v)
    if how == "mean":
        Wv = W.multiply(valid[None, :]).tocsr()
        den = np.asarray(Wv.sum(axis=1)).ravel()
        num = Wv @ np.nan_to_num(v)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)
    if how == "max":
        out = np.full(W.shape[0], np.nan)
        Wc = W.tocsr()
        for d in range(W.shape[0]):
            cols = Wc.indices[Wc.indptr[d]:Wc.indptr[d + 1]]
            vals = v[cols]; vals = vals[np.isfinite(vals)]
            if len(vals):
                out[d] = vals.max()
        return out
    raise ValueError(how)


def alert_level(p_heavy, p_vheavy, p_xheavy, thr=(0.4, 0.4, 0.4)) -> np.ndarray:
    """IMD-style colour from exceedance probabilities (thresholds tunable by the issuing authority)."""
    lvl = np.full(len(p_heavy), "Green", dtype=object)
    lvl[p_heavy >= thr[0]] = "Yellow"
    lvl[p_vheavy >= thr[1]] = "Orange"
    lvl[p_xheavy >= thr[2]] = "Red"
    return lvl


def district_table(districts: gpd.GeoDataFrame, W, raw, corr, obs, probs) -> pd.DataFrame:
    """One row per district for one day. probs: (3, lat, lon) exceedance probabilities."""
    t = districts[["district_id", "district", "state"]].copy()
    t["raw_mean"] = aggregate(raw, W, "mean"); t["raw_max"] = aggregate(raw, W, "max")
    t["corr_mean"] = aggregate(corr, W, "mean"); t["corr_max"] = aggregate(corr, W, "max")
    if obs is not None:
        t["obs_mean"] = aggregate(obs, W, "mean"); t["obs_max"] = aggregate(obs, W, "max")
    t["p_heavy"] = aggregate(probs[0], W, "max"); t["p_vheavy"] = aggregate(probs[1], W, "max"); t["p_xheavy"] = aggregate(probs[2], W, "max")
    t["alert"] = alert_level(t["p_heavy"].fillna(0).values, t["p_vheavy"].fillna(0).values, t["p_xheavy"].fillna(0).values)
    return t
