"""
Train everything and write predictions for the test season.

  Stage A  regime classifier on synoptic channels (labels from 03_labels.py)
  Stage B  shared U-Net on all training days (warm start), then 4 regime experts
           fine-tuned on their own labelled days; hard-routed at test time by the
           classifier's argmax.
Outputs (data/processed/):
  models/classifier.pt, models/unet_shared.pt, models/unet_<regime>.pt
  predictions.nc      raw / corrected (shared) / corrected (experts) / obs / probs for every test sample
  classifier_eval.json, train_history.json
"""
import _common  # noqa: F401
import json, time
from pathlib import Path
import numpy as np, pandas as pd, torch, xarray as xr

from rainfall.config import load_config
from rainfall.data.build_dataset import ChannelNormalizer, CHANNELS, SYNOPTIC
from rainfall.regimes.labels import CLASSES
from rainfall.regimes.classifier import train_classifier, predict_proba, evaluate
from rainfall.correction.unet import UNet, make_expert, train_unet, predict, EXPERT_WEIGHT_DECAY
from rainfall.grid import LAT, LON


def main(epochs_shared=40, epochs_expert=15, base=32):
    cfg = load_config(); out = Path(cfg["paths"]["processed"]); mdir = out / "models"; mdir.mkdir(exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")
    ds = xr.open_zarr(out / "dataset.zarr")
    X = ds["X"].values; y = ds["y"].values; split = ds["split"].values
    labels = pd.read_csv(out / "regime_labels.csv", index_col=0)
    lab = labels["label"].values.astype(int)
    norm = ChannelNormalizer.load(out / "norm_stats.json"); Xn = norm.transform(X)

    # time-ordered split: last 15% of training days = validation (early stopping); never random
    tr_idx = np.where(split == "train")[0]; te_idx = np.where(split == "test")[0]
    n_val = max(int(0.15 * len(tr_idx)), 8)
    va_idx, tr_idx = tr_idx[-n_val:], tr_idx[:-n_val]
    print(f"train {len(tr_idx)}  val {len(va_idx)}  test {len(te_idx)}")
    hist = {}

    # ---------------- Stage A: classifier
    syn = [CHANNELS.index(c) for c in SYNOPTIC]
    t = time.time()
    clf, h = train_classifier(Xn[tr_idx][:, syn], lab[tr_idx], Xn[va_idx][:, syn], lab[va_idx], epochs=80, device=dev)
    hist["classifier"] = h
    proba_te = predict_proba(clf, Xn[te_idx][:, syn], dev)
    ev = evaluate(lab[te_idx], proba_te)
    print(f"classifier: {time.time() - t:.0f}s  test acc {ev['accuracy']:.3f}")
    print(pd.DataFrame(ev["confusion"], index=CLASSES, columns=CLASSES))
    json.dump(ev, open(out / "classifier_eval.json", "w"), indent=1)
    torch.save(clf.state_dict(), mdir / "classifier.pt")

    # ---------------- Stage B1: shared U-Net (warm start)
    t = time.time()
    shared = UNet(in_ch=len(CHANNELS), base=base)
    shared, h = train_unet(shared, Xn[tr_idx], y[tr_idx], Xn[va_idx], y[va_idx], epochs=epochs_shared, device=dev)
    hist["unet_shared"] = h; torch.save(shared.state_dict(), mdir / "unet_shared.pt")
    print(f"shared U-Net: {time.time() - t:.0f}s")
    rain_shared, prob_shared = predict(shared, Xn[te_idx], dev)

    # ---------------- Stage B2: regime experts, fine-tuned from the shared weights on their own days
    experts = {}
    for k, regime in enumerate(CLASSES):
        idx = tr_idx[lab[tr_idx] == k]; vidx = va_idx[lab[va_idx] == k]
        if len(vidx) < 3:
            vidx = va_idx  # too few of this regime in validation - use all
        exp = make_expert(regime, base=base, in_ch=len(CHANNELS))
        # warm start: copy every weight whose shape matches (7x7 encoder convs differ from 3x3 -> trained from scratch)
        sd = shared.state_dict(); own = exp.state_dict()
        exp.load_state_dict({n: (sd[n] if n in sd and sd[n].shape == own[n].shape else own[n]) for n in own})
        if len(idx) < 5:
            print(f"expert {regime}: only {len(idx)} training days -> using shared weights"); experts[regime] = shared; continue
        t = time.time()
        exp, h = train_unet(exp, Xn[idx], y[idx], Xn[vidx], y[vidx], epochs=epochs_expert, lr=3e-4,
                            weight_decay=EXPERT_WEIGHT_DECAY[regime], device=dev, patience=5)
        hist[f"unet_{regime}"] = h; experts[regime] = exp
        torch.save(exp.state_dict(), mdir / f"unet_{regime.replace('/', '_')}.pt")
        print(f"expert {regime}: {len(idx)} days, {time.time() - t:.0f}s")

    # ---------------- hard routing on the test season
    route = proba_te.argmax(1)
    rain_exp = np.zeros_like(rain_shared); prob_exp = np.zeros_like(prob_shared)
    for k, regime in enumerate(CLASSES):
        sel = np.where(route == k)[0]
        if len(sel):
            r, p = predict(experts[regime], Xn[te_idx][sel], dev); rain_exp[sel] = r; prob_exp[sel] = p

    pred = xr.Dataset(
        {"raw": (("sample", "lat", "lon"), X[te_idx, 0]),
         "corrected_shared": (("sample", "lat", "lon"), rain_shared),
         "corrected_experts": (("sample", "lat", "lon"), rain_exp),
         "obs": (("sample", "lat", "lon"), y[te_idx]),
         "prob_shared": (("sample", "thr", "lat", "lon"), prob_shared),
         "prob_experts": (("sample", "thr", "lat", "lon"), prob_exp),
         "regime_proba": (("sample", "regime"), proba_te)},
        coords={"sample": ds["sample"].values[te_idx], "lat": LAT, "lon": LON, "thr": [64.5, 115.5, 204.4],
                "regime": CLASSES, "valid_date": ("sample", ds["valid_date"].values[te_idx]),
                "init": ("sample", ds["init"].values[te_idx]), "lead": ("sample", ds["lead"].values[te_idx]),
                "regime_label": ("sample", lab[te_idx]), "regime_pred": ("sample", route)})
    pred.to_netcdf(out / "predictions.nc")
    json.dump(hist, open(out / "train_history.json", "w"))
    print("saved", out / "predictions.nc")


if __name__ == "__main__":
    main()
