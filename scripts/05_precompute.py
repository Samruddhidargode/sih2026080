"""
Precompute everything the dashboard reads (so the app never runs a model):
  district_daily.parquet   one row per (date, district): raw/corr/obs mean & max, probs, alert
  metrics_daily.csv        per-day RMSE / POD / FAR / CSI @ 64.5 for raw & corrected
  metrics_season.csv       season table: all metrics x thresholds x forecasts (+ side-by-side)
  regime_daily.csv         predicted regime + probabilities + label per day
  districts.geojson        simplified boundaries
"""
import _common  # noqa: F401
import json
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr

from rainfall.config import load_config
from rainfall.data.districts import load_districts
from rainfall.aggregate.districts import build_weights, district_table
from rainfall.verify.metrics import score_table, side_by_side, contingency_scores, rmse, brier
from rainfall.regimes.labels import CLASSES


def main():
    cfg = load_config(); out = Path(cfg["paths"]["processed"])
    pred = xr.open_dataset(out / "predictions.nc")
    dates = pd.to_datetime(pred["valid_date"].values)

    # ---- which corrected forecast is the headline? keep both, flag the better by CSI@64.5
    obs = pred["obs"].values
    fc = {"Raw NWP": pred["raw"].values, "Corrected (regime experts)": pred["corrected_experts"].values,
          "Corrected (shared U-Net)": pred["corrected_shared"].values}
    season = score_table(fc, obs)
    season.to_csv(out / "metrics_season.csv", index=False)
    csi = season[(season.metric == "CSI") & (season.threshold.str.startswith("heavy"))].set_index("forecast")["value"]
    best = "experts" if csi["Corrected (regime experts)"] >= csi["Corrected (shared U-Net)"] else "shared"
    print("season CSI@64.5:", csi.round(3).to_dict(), "-> headline:", best)
    for name in ["experts", "shared"]:
        sbs = side_by_side(season, raw="Raw NWP", corr=f"Corrected ({'regime experts' if name == 'experts' else 'shared U-Net'})")
        sbs.to_csv(out / f"metrics_side_by_side_{name}.csv", index=False)
    probs = pred[f"prob_{best}"].values
    bs = {f"brier_{t}": brier(probs[:, i], obs, t) for i, t in enumerate([64.5, 115.5, 204.4])}
    json.dump({"headline": best, **bs}, open(out / "headline.json", "w"), indent=1)

    # ---- per-day metrics
    rows = []
    for i, d in enumerate(dates):
        for name, arr in fc.items():
            c = contingency_scores(arr[i:i + 1], obs[i:i + 1], 64.5)
            rows.append(dict(date=d, forecast=name, rmse=rmse(arr[i:i + 1], obs[i:i + 1]), **{k: c[k] for k in ("POD", "FAR", "CSI", "ETS", "hits", "misses", "false_alarms")},
                             obs_heavy_cells=int(np.nansum(obs[i] > 64.5)), obs_max=float(np.nanmax(obs[i]))))
    pd.DataFrame(rows).to_csv(out / "metrics_daily.csv", index=False)

    # ---- regimes
    rp = pred["regime_proba"].values
    reg = pd.DataFrame({"date": dates, "regime_pred": [CLASSES[k] for k in pred["regime_pred"].values],
                        "regime_label": [CLASSES[k] for k in pred["regime_label"].values],
                        **{f"p_{c}": rp[:, k] for k, c in enumerate(CLASSES)}})
    reg.to_csv(out / "regime_daily.csv", index=False)

    # ---- districts
    gdf = load_districts(cfg, simplify_tolerance_deg=0.01)
    W = build_weights(gdf, out / "district_weights.npz")
    gdf.to_file(out / "districts.geojson", driver="GeoJSON")
    corr = pred[f"corrected_{best}"].values
    tabs = []
    for i, d in enumerate(dates):
        t = district_table(gdf, W, pred["raw"].values[i], corr[i], obs[i], probs[i]); t.insert(0, "date", d); tabs.append(t)
    pd.concat(tabs).to_parquet(out / "district_daily.parquet", index=False)
    print("done:", len(dates), "days,", len(gdf), "districts")


if __name__ == "__main__":
    main()
