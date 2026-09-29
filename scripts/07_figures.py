"""Static PNG figures for the slides -> docs/figures/"""
import _common  # noqa: F401
import json
from pathlib import Path
import numpy as np, pandas as pd, xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from rainfall.config import load_config
from rainfall.regimes.labels import CLASSES

REG_COL = {"Active": "#1f77b4", "Break": "#ff7f0e", "Depression": "#d62728", "Coastal/Orographic": "#2ca02c"}
LEVELS = [0, 2.5, 7.5, 15.5, 35.5, 64.5, 115.5, 204.4, 400]
CMAP = ListedColormap(["#ffffff", "#d4ecf7", "#7cc4e8", "#2b8cbe", "#31a354", "#ffd92f", "#fd8d3c", "#e31a1c"])


def rain_ax(ax, arr, land, lon, lat, title):
    a = np.where(land > 0.05, arr, np.nan)
    im = ax.pcolormesh(lon, lat, a, cmap=CMAP, norm=BoundaryNorm(LEVELS, CMAP.N), shading="nearest")
    ax.contour(lon, lat, land, levels=[0.5], colors="k", linewidths=0.5)
    ax.set_title(title, fontsize=12); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    return im


def main():
    cfg = load_config(); P = Path(cfg["paths"]["processed"]); out = Path("docs/figures"); out.mkdir(parents=True, exist_ok=True)
    pred = xr.open_dataset(P / "predictions.nc").load(); head = json.load(open(P / "headline.json"))["headline"]
    terr = xr.open_dataset(Path(cfg["paths"]["interim"]) / "terrain.nc").load()
    land, lon, lat = terr["land_frac"].values, pred["lon"].values, pred["lat"].values
    dates = pd.to_datetime(pred["valid_date"].values)
    reg = pd.read_csv(P / "regime_daily.csv", parse_dates=["date"]); mday = pd.read_csv(P / "metrics_daily.csv", parse_dates=["date"])

    # 1. case days: top 3 by observed heavy-rain extent
    heavy = [np.nansum(pred["obs"].values[i] > 64.5) for i in range(len(dates))]
    for rank, i in enumerate(np.argsort(heavy)[::-1][:3]):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5.2))
        for ax, (v, t) in zip(axs, [("raw", "Raw GFS Day-1"), (f"corrected_{head}", "Corrected (regime-aware)"), ("obs", "Observed (IMD)")]):
            im = rain_ax(ax, pred[v].values[i], land, lon, lat, t)
        cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02); cb.set_label("24-h rainfall (mm)")
        r = reg.iloc[i]
        fig.suptitle(f"{dates[i]:%d %B %Y}  ·  regime: {r.regime_pred} (p = {r[f'p_{r.regime_pred}']:.2f})", fontsize=14)
        fig.savefig(out / f"case_{rank + 1}_{dates[i]:%Y%m%d}.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    # 2. terrain
    fig, ax = plt.subplots(figsize=(6, 5.5))
    e = np.where(land > 0.05, terr["elev_mean"].values, np.nan)
    im = ax.pcolormesh(lon, lat, e, cmap="terrain", vmin=0, vmax=4500, shading="nearest"); ax.contour(lon, lat, land, levels=[0.5], colors="k", linewidths=0.5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title("Block-averaged elevation, 0.25° (SRTM 90 m)")
    fig.colorbar(im, ax=ax, fraction=0.04, label="m"); fig.savefig(out / "terrain.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    # 3. regime timeline
    fig, ax = plt.subplots(figsize=(14, 2.2))
    for d, r in zip(reg.date, reg.regime_pred):
        ax.bar(d, 1, width=1, color=REG_COL[r], edgecolor="none")
    ax.scatter(reg.date, [1.12] * len(reg), c=[REG_COL[r] for r in reg.regime_label], s=18, marker="s")
    ax.set_ylim(0, 1.3); ax.set_yticks([]); ax.set_title("Predicted regime (bars) vs observation-derived label (squares) — test season", fontsize=11)
    for k, (n, c) in enumerate(REG_COL.items()):
        ax.bar([], [], color=c, label=n)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.55), frameon=False); fig.savefig(out / "regime_timeline.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    # 4. season skill bar chart at 64.5 mm
    sbs = pd.read_csv(P / f"metrics_side_by_side_{head}.csv"); corr_col = [c for c in sbs.columns if c.startswith("Corrected")][0]
    h = sbs[(sbs.threshold == "heavy (>64.5 mm)") & (sbs.metric.isin(["POD", "FAR", "CSI", "ETS", "FSS 3x3", "FSS 5x5"]))]
    fig, ax = plt.subplots(figsize=(8, 4)); x = np.arange(len(h))
    ax.bar(x - 0.18, h["Raw NWP"], 0.36, label="Raw GFS", color="#999"); ax.bar(x + 0.18, h[corr_col], 0.36, label="Corrected", color="#1f77b4")
    ax.set_xticks(x); ax.set_xticklabels(h.metric); ax.set_ylim(0, 1); ax.legend(); ax.set_title("Skill at the heavy-rain threshold (> 64.5 mm), test season Day-1")
    for xi, (a, b) in enumerate(zip(h["Raw NWP"], h[corr_col])):
        ax.text(xi - 0.18, a + 0.01, f"{a:.2f}", ha="center", fontsize=8); ax.text(xi + 0.18, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
    fig.savefig(out / "skill_heavy.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    # 5. classifier confusion matrix
    ce = json.load(open(P / "classifier_eval.json")); cm = np.array(ce["confusion"])
    fig, ax = plt.subplots(figsize=(5, 4.3)); ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=8); ax.set_yticklabels(CLASSES, fontsize=8)
    for a in range(4):
        for b in range(4):
            ax.text(b, a, cm[a, b], ha="center", va="center", color="white" if cm[a, b] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("label from observations"); ax.set_title(f"Regime classifier, test season (acc {ce['accuracy']:.2f})")
    fig.savefig(out / "classifier_confusion.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    print("figures ->", out)


if __name__ == "__main__":
    main()
