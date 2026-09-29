"""
Stage 2a - Regime labels from OBSERVATIONS (used only to train the classifier).

Active / Break  : Rajeevan, Gadgil & Bhate (2010). Area-averaged daily rainfall
                  over the core monsoon zone (18-28N, 65-88E, land only) ->
                  standardised anomaly vs a long-period daily climatology.
                  > +1 sigma for >= 3 consecutive days -> Active
                  < -1 sigma for >= 3 consecutive days -> Break
Depression      : closed low in the (forecast) MSLP field: minimum <= `mslp_max_hpa`
                  and at least `mslp_depth_hpa` deeper than the surrounding 5x5 deg mean.
Coastal/Oro     : everything else (a JJAS day that is neither active, break nor
                  depression is dominated by Western Ghats / NE-hill / coastal rain).

Precedence when several apply: Depression > Active > Break > Coastal/Orographic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

CLASSES = ["Active", "Break", "Depression", "Coastal/Orographic"]


# ---------- Rajeevan index -------------------------------------------------

def core_zone_series(rain: xr.DataArray, cz: dict) -> pd.Series:
    """Daily area-mean rainfall (mm) over the core monsoon zone, land cells only (NaN = ocean/outside IMD)."""
    box = rain.sel(lat=slice(cz["lat_min"], cz["lat_max"]), lon=slice(cz["lon_min"], cz["lon_max"]))
    s = box.mean(["lat", "lon"], skipna=True).to_series()
    s.index = pd.DatetimeIndex(s.index)
    return s


def daily_climatology(cz_series: pd.Series, smooth_days: int = 15) -> pd.DataFrame:
    """Mean and std of core-zone rainfall for each calendar day, smoothed with a
    running window (raw daily climatology from ~30 years is noisy)."""
    df = pd.DataFrame({"r": cz_series})
    df["doy"] = df.index.dayofyear
    clim = df.groupby("doy")["r"].agg(["mean", "std"])
    clim = clim.reindex(range(1, 367)).interpolate(limit_direction="both")
    # circular smoothing so day 1 and day 366 share a window
    ext = pd.concat([clim.iloc[-smooth_days:], clim, clim.iloc[:smooth_days]])
    sm = ext.rolling(smooth_days, center=True, min_periods=1).mean().iloc[smooth_days:-smooth_days]
    sm.index = clim.index
    return sm


def standardised_anomaly(cz_series: pd.Series, clim: pd.DataFrame) -> pd.Series:
    doy = cz_series.index.dayofyear
    mu = clim.loc[doy, "mean"].values
    sd = clim.loc[doy, "std"].values
    return pd.Series((cz_series.values - mu) / sd, index=cz_series.index, name="anomaly")


def _runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    """True where `mask` is part of a run of >= min_len consecutive True values."""
    out = np.zeros_like(mask, dtype=bool)
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def active_break(anomaly: pd.Series, sigma: float = 1.0, min_run: int = 3) -> pd.Series:
    """'Active' / 'Break' / None per day. Runs are evaluated within each year separately."""
    lab = pd.Series([None] * len(anomaly), index=anomaly.index, dtype=object)
    for _, a in anomaly.groupby(anomaly.index.year):
        act = _runs((a.values > sigma), min_run)
        brk = _runs((a.values < -sigma), min_run)
        lab.loc[a.index[act]] = "Active"
        lab.loc[a.index[brk]] = "Break"
    return lab


# ---------- Depression from MSLP ----------------------------------------------

def depression_from_mslp(mslp: xr.DataArray, mslp_max_hpa: float = 1000.0, depth_hpa: float = 4.0,
                         region: dict | None = None) -> tuple[bool, dict]:
    """
    Closed-low test on one (lat, lon) MSLP field in hPa.
    Returns (is_depression, info) with the minimum's location and depth.
    Region defaults to the Bay of Bengal / monsoon-trough belt where monsoon lows form.
    """
    region = region or dict(lat_min=8.0, lat_max=28.0, lon_min=68.0, lon_max=95.0)
    sub = mslp.sel(lat=slice(region["lat_min"], region["lat_max"]), lon=slice(region["lon_min"], region["lon_max"]))
    vals = sub.values
    if np.all(np.isnan(vals)):
        return False, {}
    k = np.nanargmin(vals)
    iy, ix = np.unravel_index(k, vals.shape)
    pmin = float(vals[iy, ix])
    lat0, lon0 = float(sub.lat[iy]), float(sub.lon[ix])
    ring = mslp.sel(lat=slice(lat0 - 2.5, lat0 + 2.5), lon=slice(lon0 - 2.5, lon0 + 2.5))
    depth = float(np.nanmean(ring.values) - pmin)
    info = dict(mslp_min=pmin, depth=depth, lat=lat0, lon=lon0)
    return bool(pmin <= mslp_max_hpa and depth >= depth_hpa), info


# ---------- assemble ----------------------------------------------------------

def label_days(obs: xr.DataArray, clim_obs: xr.DataArray, mslp_by_date: dict, cfg: dict) -> pd.DataFrame:
    """
    obs        : IMD rain for the years to label (time, lat, lon)
    clim_obs   : IMD rain for the climatology period (time, lat, lon)
    mslp_by_date: {date -> (lat, lon) MSLP DataArray in hPa} for depression detection
    Returns DataFrame indexed by date: anomaly, active_break, depression, regime.
    """
    rc = cfg["regimes"]
    cz = rc["core_zone"]
    clim = daily_climatology(core_zone_series(clim_obs, cz))
    anom = standardised_anomaly(core_zone_series(obs, cz), clim)
    ab = active_break(anom, rc["anomaly_sigma"], rc["min_run_days"])
    df = pd.DataFrame({"anomaly": anom, "active_break": ab})
    dep, depinfo = [], []
    for d in df.index:
        m = mslp_by_date.get(pd.Timestamp(d).normalize())
        if m is None:
            dep.append(False); depinfo.append({})
            continue
        ok, info = depression_from_mslp(m, rc.get("mslp_max_hpa", 1000.0), rc.get("mslp_depth_hpa", 4.0))
        dep.append(ok); depinfo.append(info)
    df["depression"] = dep
    df["mslp_min"] = [i.get("mslp_min", np.nan) for i in depinfo]
    df["mslp_depth"] = [i.get("depth", np.nan) for i in depinfo]
    df["regime"] = np.where(df["depression"], "Depression",
                   np.where(df["active_break"] == "Active", "Active",
                   np.where(df["active_break"] == "Break", "Break", "Coastal/Orographic")))
    return df
