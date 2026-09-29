"""
Operational run: today's (or any) 00Z GFS -> regime -> routed U-Net -> district table.

    python scripts/06_predict_today.py                 # today 00Z, Day-1
    python scripts/06_predict_today.py --init 2026-09-20 --lead 1
Writes data/live/<valid_date>/{grid.nc, districts.csv, regime.json} and data/live/latest.json
"""
import _common  # noqa: F401
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch, xarray as xr

from rainfall.config import load_config
from rainfall.data.gfs import build_gfs_sample, valid_date
from rainfall.data.dem import load_terrain
from rainfall.data.build_dataset import ChannelNormalizer, CHANNELS, DYNAMIC, SYNOPTIC
from rainfall.data.districts import load_districts
from rainfall.aggregate.districts import build_weights, district_table
from rainfall.regimes.classifier import RegimeCNN
from rainfall.regimes.labels import CLASSES
from rainfall.correction.unet import UNet, make_expert, predict
from rainfall.grid import LAT, LON


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
    ap.add_argument("--lead", type=int, default=1)
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config(); out = Path(cfg["paths"]["processed"]); mdir = out / "models"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    init = pd.Timestamp(a.init); vd = valid_date(init, a.lead)
    print(f"GFS {init:%Y-%m-%d} 00Z, Day-{a.lead} -> valid {vd:%Y-%m-%d}")

    g = build_gfs_sample(init, a.lead, cfg)                       # download + align (cached)
    terr = load_terrain(cfg)
    X = np.concatenate([np.stack([g[v].values for v in DYNAMIC]),
                        np.nan_to_num(terr["elev_mean"].values)[None], terr["land_frac"].values[None]]).astype("float32")[None]
    Xn = ChannelNormalizer.load(out / "norm_stats.json").transform(X)
    print(f"  ingest + align: {time.time() - t0:.0f} s")

    clf = RegimeCNN(); clf.load_state_dict(torch.load(mdir / "classifier.pt", map_location="cpu")); clf.eval()
    syn = [CHANNELS.index(c) for c in SYNOPTIC]
    with torch.no_grad():
        proba = torch.softmax(clf(torch.tensor(Xn[:, syn])), 1).numpy()[0]
    regime = CLASSES[int(proba.argmax())]
    print("  regime:", regime, {c: round(float(p), 2) for c, p in zip(CLASSES, proba)})

    head = json.load(open(out / "headline.json"))["headline"]
    if head == "experts" and (mdir / f"unet_{regime.replace('/', '_')}.pt").exists():
        net = make_expert(regime); net.load_state_dict(torch.load(mdir / f"unet_{regime.replace('/', '_')}.pt", map_location="cpu"))
    else:
        net = UNet(); net.load_state_dict(torch.load(mdir / "unet_shared.pt", map_location="cpu"))
    rain, prob = predict(net.to(dev), Xn, dev); rain, prob = rain[0], prob[0]

    gdf = load_districts(cfg, simplify_tolerance_deg=0.01); W = build_weights(gdf, out / "district_weights.npz")
    tab = district_table(gdf, W, X[0, 0], rain, None, prob)
    ldir = Path(cfg["paths"]["processed"]).parent / "live" / f"{vd:%Y-%m-%d}"; ldir.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"raw": (("lat", "lon"), X[0, 0]), "corrected": (("lat", "lon"), rain), "prob": (("thr", "lat", "lon"), prob)},
               coords={"lat": LAT, "lon": LON, "thr": [64.5, 115.5, 204.4]}).to_netcdf(ldir / "grid.nc")
    tab.to_csv(ldir / "districts.csv", index=False)
    info = dict(init=str(init.date()), lead=a.lead, valid_date=str(vd.date()), regime=regime,
                proba={c: float(p) for c, p in zip(CLASSES, proba)}, seconds=round(time.time() - t0, 1),
                n_orange_red=int(tab.alert.isin(["Orange", "Red"]).sum()))
    json.dump(info, open(ldir / "regime.json", "w"), indent=1); json.dump(info, open(ldir.parent / "latest.json", "w"), indent=1)
    print(f"  done in {info['seconds']} s -> {ldir}   Orange/Red districts: {info['n_orange_red']}")
    print(tab[tab.alert.isin(['Orange', 'Red'])].sort_values('corr_max', ascending=False)[['district', 'state', 'raw_max', 'corr_max', 'p_heavy', 'p_vheavy', 'alert']].head(10).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
