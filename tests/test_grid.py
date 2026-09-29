import numpy as np, xarray as xr
from rainfall.grid import LAT, LON, NLAT, NLON, to_master_grid, is_on_master_grid


def test_master_grid_shape():
    assert (NLAT, NLON) == (129, 135)
    assert LAT[0] == 6.5 and LAT[-1] == 38.5 and LON[0] == 66.5 and LON[-1] == 100.0


def test_exact_selection_from_gfs_style_global_grid():
    lat = np.arange(90, -90.01, -0.25); lon = np.arange(0, 360, 0.25)
    vals = lat[:, None] * 1000 + lon[None, :]
    da = xr.DataArray(vals, coords={"latitude": lat, "longitude": lon}, dims=("latitude", "longitude"))
    out = to_master_grid(da)
    assert out.shape == (129, 135) and is_on_master_grid(out)
    assert float(out.sel(lat=20.0, lon=80.0)) == 20.0 * 1000 + 80.0


def test_bilinear_from_finer_grid():
    lat = np.arange(0, 45.01, 0.1); lon = np.arange(60, 105.01, 0.1)
    da = xr.DataArray(lat[:, None] + lon[None, :] * 0, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
    out = to_master_grid(da)
    assert np.allclose(out.sel(lon=80.0).values, LAT, atol=1e-6)
