"""
Physics layer: terrain / coast geometry, the per-box local rain type (Layer 2),
day-level synoptic indices (inputs to the Layer-1 classifier) and the pixel
features used by the LightGBM expert. Everything here is computed from the
FORECAST fields, so it is available at forecast time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, maximum_filter, uniform_filter

KM_PER_DEG = 111.2


def static_layers(X: np.ndarray, lat: np.ndarray) -> dict:
    elev = np.nan_to_num(X[0, 4]); land = np.nan_to_num(X[0, 5])
    e = uniform_filter(elev, 3)                                   # smooth to the ~25-75 km scale winds represent
    dy = 0.25 * KM_PER_DEG
    dx = 0.25 * KM_PER_DEG * np.cos(np.deg2rad(lat))[:, None]
    dhdy = np.gradient(e, axis=0) / dy                            # m per km, +north
    dhdx = np.gradient(e, axis=1) / dx                            # m per km, +east
    ls = uniform_filter(land.astype("float64"), 5)
    ny = np.gradient(ls, axis=0) / dy; nx = np.gradient(ls, axis=1) / dx
    nn = np.hypot(nx, ny) + 1e-9
    sea = land < 0.5
    coast = binary_dilation(sea, iterations=2) & ~sea             # land boxes within ~50 km of the sea
    return dict(elev=elev, land=land, dhdx=dhdx, dhdy=dhdy, slope=np.hypot(dhdx, dhdy),
                nx=nx / nn, ny=ny / nn, coast=coast.astype("float32"))


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class LocalLayer:
    """Layer 2: orographic / coastal / plains probability for every box and day."""

    def fit(self, X, st, domain):
        oro, con = self.strengths(X, st)
        steep = st["slope"] > np.percentile(st["slope"][domain], 75)
        self.t_oro = float(np.percentile(oro[:, domain & steep], 85))
        cz = domain & (st["coast"] > 0)
        pos = con[:, cz]
        self.t_con = float(np.percentile(pos[pos > 0], 50))
        self.s_ref = float(np.percentile(st["slope"][domain], 75))
        return self

    @staticmethod
    def strengths(X, st):
        u, v = X[:, 1], X[:, 2]
        oro = np.maximum(u * st["dhdx"] + v * st["dhdy"], 0)                   # moist air forced up the slope
        con = np.maximum(u * st["nx"] + v * st["ny"], 0) * st["coast"]         # wind blowing from sea onto land
        return oro.astype("float32"), con.astype("float32")

    def probs(self, X, st):
        oro, con = self.strengths(X, st)
        a = _sig((oro - self.t_oro) / (0.3 * self.t_oro)) * np.minimum(1, st["slope"] / self.s_ref)
        b = _sig((con - self.t_con) / (0.3 * self.t_con)) * st["coast"]
        p_oro = a * (1 - b / 2); p_coast = b * (1 - a / 2)
        p_pl = np.clip(1 - p_oro - p_coast, 0, 1)
        P = np.stack([p_oro, p_coast, p_pl]).astype("float32")
        return P / P.sum(0, keepdims=True), oro, con


def _sel(lat, lon, la0, la1, lo0, lo1):
    return np.ix_((lat >= la0) & (lat <= la1), (lon >= lo0) & (lon <= lo1))


def day_indices(X, lat, lon, domain, lab, dates) -> pd.DataFrame:
    """Large-scale forecast indices per day (what a forecaster looks at on the chart)."""
    raw, u, v, p = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    cmz = _sel(lat, lon, 18, 28, 65, 88); cmz_land = domain[cmz]
    ci = _sel(lat, lon, 15, 25, 72, 85); asea = _sel(lat, lon, 10, 18, 66.5, 74)
    bay = _sel(lat, lon, 15, 27, 75, 92)
    la_b = lat[(lat >= 15) & (lat <= 27)]; lo_b = lon[(lon >= 75) & (lon <= 92)]
    tro = _sel(lat, lon, 15, 32, 75, 85); la_t = lat[(lat >= 15) & (lat <= 32)]
    dyv = 0.25 * KM_PER_DEG * 1e3
    dxv = (0.25 * KM_PER_DEG * 1e3 * np.cos(np.deg2rad(lat)))[None, :, None]
    vort = (np.gradient(v, axis=2) / dxv - np.gradient(u, axis=1) / dyv) * 1e5   # 1e-5 s^-1
    rows = []
    # yesterday's OBSERVED core-zone anomaly is known at issue time (persistence clue)
    prev = lab["anomaly"].shift(1)
    gap = pd.Series(dates, index=dates).diff() != pd.Timedelta(days=1)
    prev[gap.values] = np.nan
    for i in range(len(X)):
        pb = p[i][bay]; k = np.unravel_index(np.argmin(pb), pb.shape)
        rows.append(dict(
            cmz_rain=float(raw[i][cmz][cmz_land].mean()),
            u_central=float(u[i][ci].mean()), v_central=float(v[i][ci].mean()),
            u_arabian=float(u[i][asea].mean()),
            mslp_min=float(pb.min()), mslp_depth=float(pb.mean() - pb.min()),
            low_lat=float(la_b[k[0]]), low_lon=float(lo_b[k[1]]),
            trough_lat=float(la_t[np.argmin(p[i][tro].mean(1))]),
            vort_max=float(np.percentile(vort[i][bay], 99)),
            anom_prev=float(prev.iloc[i]) if np.isfinite(prev.iloc[i]) else np.nan,
            doy_sin=np.sin(2 * np.pi * dates[i].dayofyear / 365.25),
            doy_cos=np.cos(2 * np.pi * dates[i].dayofyear / 365.25)))
    return pd.DataFrame(rows, index=dates)


def pixel_features(X, st, oro, con, P_local, day_df, clim, qm_rank, e0_amt, lat, lon, domain):
    """(n_days, n_boxes) arrays for the LightGBM expert, IMD land boxes only."""
    n = len(X); idx = np.where(domain.ravel())[0]
    raw = X[:, 0]
    LA, LO = np.meshgrid(lat, lon, indexing="ij")

    def flat(a):                      # (n,H,W) or (H,W) -> (n, nb)
        a = np.broadcast_to(a, (n,) + domain.shape) if a.ndim == 2 else a
        return a.reshape(n, -1)[:, idx].astype("float32")

    def nb(fn, size):
        return np.stack([fn(raw[i], size=size) for i in range(n)])

    f = {
        "raw": flat(raw), "log_raw": flat(np.log1p(raw)),
        "nb3_mean": flat(nb(uniform_filter, 3)), "nb3_max": flat(nb(maximum_filter, 3)),
        "nb7_mean": flat(nb(uniform_filter, 7)), "nb7_max": flat(nb(maximum_filter, 7)),
        "u850": flat(X[:, 1]), "v850": flat(X[:, 2]), "wind": flat(np.hypot(X[:, 1], X[:, 2])),
        "mslp_anom": flat(X[:, 3] - X[:, 3].mean((1, 2), keepdims=True)),
        "upslope": flat(oro), "onshore": flat(con),
        "p_oro": flat(P_local[0]), "p_coast": flat(P_local[1]),
        "elev": flat(st["elev"]), "slope": flat(st["slope"]), "lat": flat(LA), "lon": flat(LO),
        "clim_mean": flat(clim["mean"]), "clim_p95": flat(clim["p95"]), "clim_wet": flat(clim["wet"]),
        "fc_rank": flat(qm_rank), "e0_amount": flat(e0_amt),
        "dist_low_km": flat(np.stack([np.hypot((LA - r.low_lat) * KM_PER_DEG,
                                               (LO - r.low_lon) * KM_PER_DEG * np.cos(np.deg2rad(LA)))
                                      for r in day_df.itertuples()])),
        "dlat_low": flat(np.stack([LA - r.low_lat for r in day_df.itertuples()])),
        "dlon_low": flat(np.stack([LO - r.low_lon for r in day_df.itertuples()])),
    }
    for c in day_df.columns:          # broadcast day indices to every box
        f[c] = np.repeat(day_df[c].values[:, None].astype("float32"), len(idx), 1)
    return f


def box_climatology(y: np.ndarray) -> dict:
    with np.errstate(all="ignore"):
        return dict(mean=np.nanmean(y, 0), p95=np.nanpercentile(y, 95, axis=0),
                    wet=np.nanmean(np.where(np.isfinite(y), (y > 2.5).astype(float), np.nan), 0))
