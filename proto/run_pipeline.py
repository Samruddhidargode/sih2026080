"""
End-to-end prototype run (all precomputed for the app):

  1. Layer 2 thresholds + Layer 1 day indices           (physics, from the forecast)
  2. Leave-one-year-out on 2023 / 2024                  -> out-of-fold predictions of all 4 experts
  3. Soft blender + calibration fitted on out-of-fold   (never on data the experts trained on)
  4. Final experts trained on 2023+2024, scored ONCE on the locked 2025 season
  5. Verification, district products, diagnostics      -> proto/output/

    .venv/Scripts/python.exe proto/run_pipeline.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr

from common import (DAY_SIT, DEV_YEARS, EXPERTS, LOCAL, OUT, PROC, SITUATIONS, TEST_YEAR, THR,
                    load_cube, truth_situation)
from features import LocalLayer, box_climatology, day_indices, pixel_features, static_layers
from regimes import DayRegimeClassifier, situation_probs
from experts import E0QuantileMap, E1RegimeQM, E2LightGBM, E3UNet
from blend import Calibrator, SoftBlender
from verify import best_threshold, block_bootstrap_csi, cat_scores, counts, season_table

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def run_fold(C, st, locP, oro, con, day_df, tr, te):
    X, y, lab, dom = C["X"], C["y"], C["lab"], C["domain"]
    raw = X[:, 0]
    clf = DayRegimeClassifier().fit(day_df[tr], lab[tr])
    pday_tr, pday_te = clf.predict(day_df[tr]), clf.predict(day_df[te])
    S_tr, S_te = situation_probs(pday_tr, locP[:, tr]), situation_probs(pday_te, locP[:, te])

    e0 = E0QuantileMap().fit(raw[tr], y[tr], dom); A0, P0 = e0.predict(raw[te]); log("   E0 done")
    e1 = E1RegimeQM(e0.qm).fit(raw[tr], y[tr], S_tr); A1, P1 = e1.predict(raw[te], S_te); log("   E1 done")
    del S_tr
    clim = box_climatology(y[tr]); idx = np.where(dom.ravel())[0]

    def feats(sel):
        return pixel_features(X[sel], st, oro[sel], con[sel], locP[:, sel], day_df[sel], clim,
                              e0.qm.rank(raw[sel], "F"), e0.qm.map(raw[sel]), C["lat"], C["lon"], dom)
    ftr = feats(tr)
    e2 = E2LightGBM().fit(ftr, y[tr].reshape(int(tr.sum()), -1)[:, idx]); del ftr
    A2, P2 = e2.predict(feats(te), dom); log("   E2 done")
    e3 = E3UNet().fit(X[tr], y[tr]); A3, P3 = e3.predict(X[te], dom); log("   E3 done")
    return dict(A=np.stack([A0, A1, A2, A3]), P=np.stack([P0, P1, P2, P3]), pday=pday_te, S=S_te,
                imp=e2.importance(), e1n=e1.n)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    C = load_cube(); X, y, dates, years, lab, dom = C["X"], C["y"], C["dates"], C["years"], C["lab"], C["domain"]
    dev, test = np.isin(years, DEV_YEARS), years == TEST_YEAR
    log(f"cube {X.shape}  dev days {dev.sum()}  test days {test.sum()}  IMD boxes {dom.sum()}")

    st = static_layers(X, C["lat"])
    loc = LocalLayer().fit(X[dev], st, dom)
    locP, oro, con = loc.probs(X, st)
    day_df = day_indices(X, C["lat"], C["lon"], dom, lab, dates)
    truth = truth_situation(lab)
    log("physics layers ready")

    # ---------------------------------------------------------------- out-of-fold on dev years
    n = len(dates)
    oA = np.full((4,) + y.shape, np.nan, "float32"); oP = np.full((4, 3) + y.shape, np.nan, "float32")
    oS = np.zeros((12,) + y.shape, "float32"); opday = pd.DataFrame(index=dates, columns=DAY_SIT, dtype=float)
    for hold in DEV_YEARS:
        tr, te = dev & (years != hold), years == hold
        log(f"fold: train {sorted(set(years[tr]))} -> predict {hold}")
        r = run_fold(C, st, locP, oro, con, day_df, tr, te)
        oA[:, te], oP[:, :, te], oS[:, te] = r["A"], r["P"], r["S"]; opday.loc[dates[te]] = r["pday"].values

    # ---------------------------------------------------------------- blender + calibration (OOF only)
    bl = SoftBlender().fit(oA[:, dev], oP[:, :, dev], oS[:, dev], y[dev])
    bA_oof, bP_oof, _ = bl.apply(oA[:, dev], oP[:, :, dev], oS[:, dev])
    cal = Calibrator().fit(bP_oof, y[dev]); bP_oof = cal.apply(bP_oof)
    names = EXPERTS + ["Regime-aware blend"]
    oof_probs = {nm: oP[k][:, dev] for k, nm in enumerate(EXPERTS)}
    oof_probs["Regime-aware blend"] = bP_oof
    pstar = {nm: {t: best_threshold(oof_probs[nm][k], y[dev], t) for k, t in enumerate(THR[:2])} for nm in names}
    log("blender + calibration fitted on out-of-fold data; p* =", {k: v[THR[0]] for k, v in pstar.items()})
    del oA, oP

    # ---------------------------------------------------------------- final models -> locked 2025 test
    log(f"final: train {DEV_YEARS} -> predict {TEST_YEAR}")
    r = run_fold(C, st, locP, oro, con, day_df, dev, test)
    A, P, S = r["A"], r["P"], r["S"]
    bA, bP, Wa = bl.apply(A, P, S); bP = cal.apply(bP)
    obs = y[test]; raw = X[test, 0]; mask = np.isfinite(obs)

    with np.errstate(all="ignore"):
        clim_f = {t: np.nanmean(np.where(np.isfinite(y[dev]), (y[dev] > t).astype(float), np.nan), 0) for t in THR[:2]}
    methods = {"Raw GFS": dict(amount=raw, yes={t: raw > t for t in THR[:2]}, prob=None)}
    for k, nm in enumerate(EXPERTS):
        methods[nm] = dict(amount=A[k], yes={t: P[k, i] >= pstar[nm][t] for i, t in enumerate(THR[:2])},
                           prob={t: P[k, i] for i, t in enumerate(THR[:2])})
    methods["Regime-aware blend"] = dict(amount=bA, yes={t: bP[i] >= pstar["Regime-aware blend"][t] for i, t in enumerate(THR[:2])},
                                         prob={t: bP[i] for i, t in enumerate(THR[:2])})
    season = pd.DataFrame(season_table(methods, obs, clim_f))
    season.to_csv(OUT / "scores_season.csv", index=False)
    log("\n" + season[["method", "RMSE", "Bias", "POD@64.5", "FAR@64.5", "CSI@64.5", "ETS@64.5", "FSS9@64.5", "BSS@64.5", "CSI@115.5"]].round(3).to_string(index=False))

    # stratified by observed day situation and by forecast local type
    ttest = truth[test].values; ltype = np.argmax(locP[:, test], 0)
    strat = []
    for g in DAY_SIT:
        m = np.broadcast_to((ttest == g)[:, None, None], obs.shape)
        if m.any():
            for row in season_table(methods, obs, clim_f, mask=m, thr_list=(64.5,)):
                strat.append(dict(group="Day: " + g, days=int((ttest == g).sum()), **row))
    for k, g in enumerate(LOCAL):
        for row in season_table(methods, obs, clim_f, mask=(ltype == k), thr_list=(64.5,)):
            strat.append(dict(group="Local: " + g, days=int(test.sum()), **row))
    pd.DataFrame(strat).to_csv(OUT / "scores_by_regime.csv", index=False)

    boot = {"blend_vs_E0": block_bootstrap_csi(methods["Regime-aware blend"]["yes"][64.5], methods["E0 Quantile map"]["yes"][64.5], obs, 64.5, mask),
            "blend_vs_raw": block_bootstrap_csi(methods["Regime-aware blend"]["yes"][64.5], methods["Raw GFS"]["yes"][64.5], obs, 64.5, mask)}
    log("bootstrap CSI@64.5:", boot)

    # ---------------------------------------------------------------- the problem: raw error by weather type (dev years)
    diag = []
    lt_dev = np.argmax(locP[:, dev], 0); tdev = truth[dev].values; od, rd = y[dev], X[dev, 0]
    for label, m in [*[(f"Day: {g}", np.broadcast_to((tdev == g)[:, None, None], od.shape)) for g in DAY_SIT],
                     *[(f"Local: {g}", lt_dev == k) for k, g in enumerate(LOCAL)]]:
        ok = m & np.isfinite(od)
        c = counts(rd > 64.5, od > 64.5, ok); s = cat_scores(c)
        diag.append(dict(group=label, box_days=int(ok.sum()), total_ratio=float(rd[ok].sum() / od[ok].sum()),
                         wet_raw=float((rd[ok] > 2.5).mean()), wet_obs=float((od[ok] > 2.5).mean()),
                         heavy_freq_bias=s["FreqBias"], POD=s["POD"], CSI=s["CSI"]))
    pd.DataFrame(diag).to_csv(OUT / "diag_problem.csv", index=False)

    # ---------------------------------------------------------------- regime classifier skill (OOF dev + test)
    def conf(pred, true):
        return pd.crosstab(pd.Series(true, name="observed"), pd.Series(pred, name="predicted")).reindex(index=DAY_SIT, columns=DAY_SIT, fill_value=0)
    pdt = r["pday"]
    ceval = {"test_accuracy": float((pdt.idxmax(1).values == ttest).mean()),
             "oof_accuracy": float((opday[dev].idxmax(1).values == truth[dev].values).mean()),
             "confusion_test": conf(pdt.idxmax(1).values, ttest).values.tolist()}

    # ---------------------------------------------------------------- day table
    ydates = dates[test]
    day = pd.DataFrame({"date": ydates, "truth": ttest, **{f"p_{c}": pdt[c].values for c in DAY_SIT}})
    day["pred"] = pdt.idxmax(1).values
    day["obs_heavy_cells"] = [int(np.nansum(obs[i] > 64.5)) for i in range(len(ydates))]
    day["obs_max"] = [float(np.nanmax(obs[i])) for i in range(len(ydates))]
    for nm, key in [("Raw GFS", "raw"), ("E0 Quantile map", "e0"), ("Regime-aware blend", "blend")]:
        day[f"csi_{key}"] = [cat_scores(counts(methods[nm]["yes"][64.5][i:i + 1], obs[i:i + 1] > 64.5, mask[i:i + 1]))["CSI"] for i in range(len(ydates))]
    for k, g in enumerate(LOCAL):
        day[f"frac_{g}"] = [float(np.mean((ltype[i] == k)[dom])) for i in range(len(ydates))]
    day = pd.concat([day, day_df.loc[ydates].reset_index(drop=True)], axis=1)
    day.to_csv(OUT / "day_table.csv", index=False)

    # ---------------------------------------------------------------- districts
    import sys; sys.path.insert(0, str(PROC.parents[1] / "src"))
    from rainfall.aggregate.districts import aggregate
    W = sp.load_npz(PROC / "district_weights.npz")
    geo = json.load(open(PROC / "districts.geojson"))
    props = pd.DataFrame([f["properties"] for f in geo["features"]])[["district_id", "district", "state"]]
    json.dump(geo, open(OUT / "districts.geojson", "w"))

    # alert cut-offs = the CSI-optimal probabilities found on OUT-OF-FOLD years (never on 2025)
    c64, c115 = pstar["Regime-aware blend"][THR[0]], pstar["Regime-aware blend"][THR[1]]

    def colour(p64, p115, p204):
        return np.select([p204 >= c115, p115 >= c115, p64 >= c64], ["Red", "Orange", "Yellow"], "Green")

    def obs_colour(mx):
        return np.select([mx >= 204.4, mx >= 115.5, mx >= 64.5], ["Red", "Orange", "Yellow"], "Green")
    rows = []
    for i, d in enumerate(ydates):
        t = props.copy(); t.insert(0, "date", d)
        for key, arr in [("raw", raw[i]), ("e0", A[0, i]), ("blend", bA[i]), ("obs", obs[i])]:
            t[f"{key}_mean"] = aggregate(arr, W, "mean"); t[f"{key}_max"] = aggregate(arr, W, "max")
        for k, t_ in enumerate(THR):
            t[f"p{int(t_)}"] = aggregate(bP[k, i], W, "max")
        t["alert"] = colour(t["p64"].fillna(0), t["p115"].fillna(0), t["p204"].fillna(0))
        t["alert_obs"] = obs_colour(t["obs_max"].fillna(0))
        t["alert_raw"] = obs_colour(t["raw_max"].fillna(0))
        rows.append(t)
    pd.concat(rows).round(3).to_csv(OUT / "district_daily.csv", index=False)
    log("districts done")

    # ---------------------------------------------------------------- fields for the app (2025)
    enc = {"zlib": True, "complevel": 4}
    ds = xr.Dataset(
        {"raw": (("date", "lat", "lon"), raw), "obs": (("date", "lat", "lon"), obs),
         **{f"amt_e{k}": (("date", "lat", "lon"), A[k]) for k in range(4)},
         "amt_blend": (("date", "lat", "lon"), bA),
         **{f"p64_e{k}": (("date", "lat", "lon"), P[k, 0]) for k in range(4)},
         "prob_blend": (("thr", "date", "lat", "lon"), bP),
         "p_local": (("local", "date", "lat", "lon"), locP[:, test]),
         "w_expert": (("expert", "date", "lat", "lon"), Wa.astype("float32"))},
        coords={"date": ydates, "lat": C["lat"], "lon": C["lon"], "thr": list(THR), "local": LOCAL, "expert": EXPERTS})
    ds["land"] = (("lat", "lon"), st["land"].astype("float32")); ds["domain"] = (("lat", "lon"), dom.astype("int8"))
    ds.to_netcdf(OUT / "fields_2025.nc", encoding={v: enc for v in ds.data_vars})

    meta = dict(pstar={k: {str(t): v for t, v in d.items()} for k, d in pstar.items()},
                weights=dict(situations=SITUATIONS, experts=EXPERTS, w_amt=bl.w_amt.round(3).tolist(),
                             w_prob=bl.w_prob.round(3).tolist(), n=bl.n,
                             global_amt=bl.global_w[0].tolist(), global_prob=bl.global_w[1].tolist()),
                bootstrap=boot, classifier=ceval, e2_importance=dict(list(r["imp"].items())[:12]),
                local_thresholds=dict(t_oro=loc.t_oro, t_con=loc.t_con),
                train_years=list(DEV_YEARS), test_year=TEST_YEAR, runtime_s=round(time.time() - T0))
    json.dump(meta, open(OUT / "meta.json", "w"), indent=1)
    log("all outputs written to", OUT)


if __name__ == "__main__":
    main()
