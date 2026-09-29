"""Shared paths, constants and data loading for the regime-aware prototype."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))          # reuse the existing U-Net / district code

PROC = ROOT / "data" / "processed"             # existing processed cube (GFS + IMD + terrain)
OUT = ROOT / "proto" / "output"

THR = (64.5, 115.5, 204.4)                     # IMD heavy / very heavy / extremely heavy (mm/day)
DEV_YEARS = (2023, 2024)                       # train + tune (leave-one-year-out)
TEST_YEAR = 2025                               # locked, used once for the final scores

DAY_SIT = ["LPS", "Active", "Break", "Normal"]  # Layer 1: large-scale regime of the day
LOCAL = ["Orographic", "Coastal", "Plains"]     # Layer 2: local rain type of each grid box
SITUATIONS = [f"{d} | {l}" for d in DAY_SIT for l in LOCAL]   # 12 combined situations
EXPERTS = ["E0 Quantile map", "E1 Regime QM", "E2 LightGBM", "E3 U-Net"]


def load_cube() -> dict:
    ds = xr.open_zarr(PROC / "dataset.zarr").load()
    X = ds["X"].values.astype("float32")        # (day, channel, lat, lon): rain, u850, v850, mslp, elev, land
    y = ds["y"].values.astype("float32")        # IMD observed 24-h rain (NaN outside India)
    dates = pd.DatetimeIndex(pd.to_datetime(ds["valid_date"].values))
    lab = pd.read_csv(PROC / "regime_labels.csv", index_col=0, parse_dates=True)
    assert (lab["sample"].values == ds["sample"].values).all(), "labels not aligned with cube"
    lab.index = dates
    domain = np.isfinite(y).any(0)             # IMD land boxes = where truth exists
    return dict(X=X, y=y, dates=dates, years=dates.year.values, lab=lab,
                lat=ds["lat"].values, lon=ds["lon"].values, domain=domain)


def truth_situation(lab: pd.DataFrame) -> pd.Series:
    """Observed day situation: LPS (depression) overrides Active / Break / Normal."""
    g1 = lab["active_break"].fillna("Normal")
    return pd.Series(np.where(lab["depression"].astype(bool), "LPS", g1), index=lab.index)
