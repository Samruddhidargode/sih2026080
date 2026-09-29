"""Regime labels for every sample in dataset.zarr -> data/processed/regime_labels.csv"""
import _common  # noqa: F401
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
from rainfall.config import load_config
from rainfall.data.imd import load_imd_rain
from rainfall.regimes.labels import label_days, CLASSES
from rainfall.grid import LAT, LON

if __name__ == "__main__":
    cfg = load_config()
    out = Path(cfg["paths"]["processed"])
    ds = xr.open_zarr(out / "dataset.zarr")
    years = sorted({pd.Timestamp(v).year for v in ds["valid_date"].values})
    obs = load_imd_rain(years, cfg["paths"]["raw"])
    clim_years = [y for y in range(1991, 2023) if (Path(cfg["paths"]["raw"]) / "imd" / "rain" / f"{y}.grd").exists()]
    print("climatology years:", clim_years[0], "-", clim_years[-1], f"({len(clim_years)} yrs)")
    clim = load_imd_rain(clim_years, cfg["paths"]["raw"])
    # MSLP for depression detection: the forecast MSLP of each sample (fxx 12 -> inside the IMD day window)
    ch = list(ds["channel"].values).index("mslp")
    mslp = {pd.Timestamp(d).normalize(): xr.DataArray(ds["X"][i, ch].values, coords={"lat": LAT, "lon": LON}, dims=("lat", "lon"))
            for i, d in enumerate(ds["valid_date"].values)}
    df = label_days(obs.sel(time=slice(f"{years[0]}-01-01", f"{years[-1]}-12-31")), clim, mslp, cfg)
    df = df.loc[[pd.Timestamp(d).normalize() for d in ds["valid_date"].values]]
    df["sample"] = ds["sample"].values; df["split"] = ds["split"].values
    df["label"] = df["regime"].map({c: i for i, c in enumerate(CLASSES)})
    df.index.name = "valid_date"
    df.to_csv(out / "regime_labels.csv")
    print(df.groupby("split")["regime"].value_counts().unstack(fill_value=0))
