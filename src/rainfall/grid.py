"""
The MASTER GRID = the IMD 0.25 deg x 0.25 deg gridded-rainfall grid.

    lat: 6.5 .. 38.5 N  step 0.25  -> 129 points
    lon: 66.5 .. 100 E  step 0.25  -> 135 points

WHY this grid and not the NWP grid: the ground truth (IMD) is what the loss
function sees, so aligning everything to the truth grid means the target never
gets interpolated (interpolating rainfall smears heavy-rain maxima - exactly
the events we care about). GFS 0.25 deg and ERA5 0.25 deg use the same
0.25-multiple coordinates, so for them "regridding" is an exact selection with
zero interpolation error. Only the DEM genuinely needs resampling.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

LAT = np.round(np.arange(6.5, 38.5 + 1e-9, 0.25), 4)    # ascending, 129
LON = np.round(np.arange(66.5, 100.0 + 1e-9, 0.25), 4)  # ascending, 135
NLAT, NLON = len(LAT), len(LON)
assert (NLAT, NLON) == (129, 135), "master grid definition drifted"

BBOX = dict(lat_min=float(LAT[0]), lat_max=float(LAT[-1]),
            lon_min=float(LON[0]), lon_max=float(LON[-1]))

_DIM_ALIASES = {
    "latitude": "lat", "Latitude": "lat", "y": "lat", "LAT": "lat",
    "longitude": "lon", "Longitude": "lon", "x": "lon", "LON": "lon",
}


def standardise_dims(da: xr.DataArray) -> xr.DataArray:
    """Rename lat/lon dims to 'lat'/'lon', convert 0..360 lon to -180..180,
    sort both ascending. Every loader calls this first so downstream code can
    assume one convention."""
    ren = {k: v for k, v in _DIM_ALIASES.items() if k in da.dims or k in da.coords}
    da = da.rename(ren)
    if "lon" not in da.dims or "lat" not in da.dims:
        raise ValueError(f"cannot find lat/lon dims in {list(da.dims)}")
    lon = da["lon"].values
    if lon.max() > 180:
        da = da.assign_coords(lon=((lon + 180) % 360) - 180)
    da = da.sortby("lon").sortby("lat")
    # float noise (e.g. 66.49999) would break exact .sel - round to the grid precision
    da = da.assign_coords(lat=np.round(da["lat"].values, 4),
                          lon=np.round(da["lon"].values, 4))
    return da


def is_on_master_grid(da: xr.DataArray, atol: float = 1e-3) -> bool:
    """True if the master lat/lon values are an exact subset of da's coords."""
    lat, lon = da["lat"].values, da["lon"].values
    return (np.all(np.isin(np.round(LAT, 3), np.round(lat, 3))) and
            np.all(np.isin(np.round(LON, 3), np.round(lon, 3))))


def to_master_grid(da: xr.DataArray, method: str = "linear") -> xr.DataArray:
    """
    Put any lat/lon DataArray on the master grid.

    - If the source already contains the master coordinates (GFS, ERA5, IMD):
      exact .sel, no interpolation.
    - Otherwise: xarray.interp. 'linear' on a regular lat/lon grid IS bilinear
      interpolation (the 4-point scheme in the design notes). We use this
      instead of xesmf because xesmf/ESMF does not install on Windows.
    """
    da = standardise_dims(da)
    if is_on_master_grid(da):
        out = da.sel(lat=LAT, lon=LON, method="nearest", tolerance=1e-3)
    else:
        # guard: refuse to extrapolate wildly outside the source domain
        if (da.lat.min() > LAT[0] + 1 or da.lat.max() < LAT[-1] - 1 or
                da.lon.min() > LON[0] + 1 or da.lon.max() < LON[-1] - 1):
            raise ValueError(
                f"source grid {float(da.lat.min())}-{float(da.lat.max())}N, "
                f"{float(da.lon.min())}-{float(da.lon.max())}E does not cover India")
        out = da.interp(lat=LAT, lon=LON, method=method)
    out = out.assign_coords(lat=LAT, lon=LON)
    return out.transpose(..., "lat", "lon")


def empty_master(fill=np.nan, dtype="float32", name: str | None = None) -> xr.DataArray:
    return xr.DataArray(np.full((NLAT, NLON), fill, dtype=dtype),
                        coords={"lat": LAT, "lon": LON}, dims=("lat", "lon"), name=name)


def cell_size_m(lat: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """(dx, dy) in metres of a 0.25 deg cell. dx shrinks with cos(lat);
    needed for terrain gradients in physical units."""
    lat = LAT if lat is None else lat
    dy = 0.25 * 111_195.0
    dx = 0.25 * 111_195.0 * np.cos(np.deg2rad(lat))
    return dx, dy
