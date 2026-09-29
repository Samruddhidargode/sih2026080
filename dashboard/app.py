"""
SIH26080 dashboard - Regime-aware post-processing of monsoon rainfall forecasts.
Reads only precomputed files from data/processed (no model runs here).

    streamlit run dashboard/app.py
"""
import json
import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xarray as xr
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
P = ROOT / "data" / "processed"

st.set_page_config(page_title="Regime-Aware Rainfall Post-Processing", layout="wide", page_icon="🌧️")

REGIME_COLORS = {"Active": "#1f77b4", "Break": "#ff7f0e", "Depression": "#d62728", "Coastal/Orographic": "#2ca02c"}
ALERT_COLORS = {"Green": "#7fbf7f", "Yellow": "#f5d442", "Orange": "#f28c28", "Red": "#d62728"}
RAIN_SCALE = [[0, "#ffffff"], [0.02, "#d4ecf7"], [0.08, "#7cc4e8"], [0.16, "#2b8cbe"], [0.28, "#0b5394"],
              [0.4, "#31a354"], [0.55, "#ffd92f"], [0.7, "#fd8d3c"], [0.85, "#e31a1c"], [1, "#67000d"]]


@st.cache_resource
def load():
    pred = xr.open_dataset(P / "predictions.nc").load()
    head = json.load(open(P / "headline.json"))
    return dict(
        pred=pred, head=head,
        dist=pd.read_parquet(P / "district_daily.parquet"),
        geo=json.load(open(P / "districts.geojson")),
        mday=pd.read_csv(P / "metrics_daily.csv", parse_dates=["date"]),
        mseason=pd.read_csv(P / "metrics_season.csv"),
        sbs=pd.read_csv(P / f"metrics_side_by_side_{head['headline']}.csv"),
        reg=pd.read_csv(P / "regime_daily.csv", parse_dates=["date"]),
        terrain=xr.open_dataset(ROOT / "data" / "interim" / "terrain.nc").load(),
        ceval=json.load(open(P / "classifier_eval.json")),
    )


D = load()
pred, head = D["pred"], D["head"]
CORR = f"corrected_{head['headline']}"; PROB = f"prob_{head['headline']}"
dates = pd.to_datetime(pred["valid_date"].values)
land = D["terrain"]["land_frac"].values

# ----------------------------------------------------------------- sidebar
st.sidebar.title("🌧️ SIH26080")
st.sidebar.caption("Regime-Aware AI Post-Processing of Monsoon Rainfall Forecasts")
page = st.sidebar.radio("View", ["Forecast day", "Live forecast (today)", "Season verification", "Regime timeline", "Case gallery", "About the system"])
sel_date = st.sidebar.selectbox("Forecast valid date (Day-1)", dates, index=int(np.argmax([np.nansum(pred["obs"].values[i] > 64.5) for i in range(len(dates))])),
                                format_func=lambda d: d.strftime("%d %b %Y"))
i = int(np.where(dates == sel_date)[0][0])
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model in use:** {'4 regime experts (hard-routed)' if head['headline'] == 'experts' else 'shared U-Net'}  \n"
                    f"Test season: **{dates.min():%b}–{dates.max():%b %Y}** ({len(dates)} days)  \nTrained on JJAS 2023–2024 · GFS 0.25° · IMD truth")


def regime_badge(i):
    r = D["reg"].iloc[i]
    c = REGIME_COLORS[r.regime_pred]
    probs = " · ".join(f"{k.replace('p_', '')} {v:.2f}" for k, v in r.items() if k.startswith("p_"))
    st.markdown(f"<div style='display:inline-block;padding:8px 16px;border-radius:8px;background:{c};color:white;font-size:20px;font-weight:700'>"
                f"Regime: {r.regime_pred}</div>&nbsp;&nbsp;<span style='color:#666'>classifier softmax → {probs}</span>"
                f"&nbsp;&nbsp;<span style='color:#999'>(label from observations: {r.regime_label})</span>", unsafe_allow_html=True)


def grid_map(arr, title, zmax=200, colorscale=RAIN_SCALE, unit="mm"):
    a = np.where(land > 0.05, arr, np.nan)  # show land only, like IMD
    fig = go.Figure(go.Heatmap(z=a, x=pred["lon"].values, y=pred["lat"].values, zmin=0, zmax=zmax, colorscale=colorscale,
                               colorbar=dict(title=unit, thickness=12), hovertemplate="%{y:.2f}N %{x:.2f}E<br>%{z:.1f} " + unit + "<extra></extra>"))
    fig.add_trace(go.Contour(z=land, x=pred["lon"].values, y=pred["lat"].values, contours=dict(start=0.5, end=0.5, size=1, coloring="none"),
                             line=dict(color="black", width=0.7), showscale=False, hoverinfo="skip"))
    fig.update_layout(title=dict(text=title, x=0.5, font=dict(size=15)), height=430, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis=dict(range=[66.5, 100], showgrid=False, title=""), yaxis=dict(range=[6.5, 38.5], scaleanchor="x", showgrid=False, title=""))
    return fig


# ----------------------------------------------------------------- pages
if page == "Forecast day":
    st.title(f"Forecast for {sel_date:%A, %d %B %Y}  ·  Day-1 from GFS 00Z {sel_date - pd.Timedelta(days=1):%d %b}")
    regime_badge(i)
    st.markdown("")
    layer = st.radio("Grid variable", ["Rainfall (mm)", "P(> 64.5 mm) heavy", "P(> 115.5 mm) very heavy", "P(> 204.4 mm) extremely heavy"], horizontal=True)
    c1, c2, c3 = st.columns(3)
    if layer.startswith("Rain"):
        c1.plotly_chart(grid_map(pred["raw"].values[i], "Raw GFS forecast"), use_container_width=True)
        c2.plotly_chart(grid_map(pred[CORR].values[i], "Corrected (regime-aware)"), use_container_width=True)
        c3.plotly_chart(grid_map(pred["obs"].values[i], "Observed (IMD)"), use_container_width=True)
    else:
        k = ["64.5", "115.5", "204.4"].index(layer.split("> ")[1].split(" ")[0])
        thr = [64.5, 115.5, 204.4][k]
        c1.plotly_chart(grid_map((pred["raw"].values[i] > thr).astype(float), f"Raw GFS: cells > {thr} mm", zmax=1, colorscale="Blues", unit=""), use_container_width=True)
        c2.plotly_chart(grid_map(pred[PROB].values[i, k], f"Corrected: P(> {thr} mm)", zmax=1, colorscale="YlOrRd", unit="prob"), use_container_width=True)
        c3.plotly_chart(grid_map((pred["obs"].values[i] > thr).astype(float), f"Observed: cells > {thr} mm", zmax=1, colorscale="Blues", unit=""), use_container_width=True)

    # ---- day metrics
    md = D["mday"][D["mday"].date == sel_date].set_index("forecast")
    st.subheader("Skill for this day (heavy rain > 64.5 mm)")
    cols = st.columns(5)
    raw, cor = md.loc["Raw NWP"], md.loc[f"Corrected ({'regime experts' if head['headline'] == 'experts' else 'shared U-Net'})"]
    for c, (name, better_low) in zip(cols, [("rmse", True), ("POD", False), ("FAR", True), ("CSI", False), ("ETS", False)]):
        delta = cor[name] - raw[name]
        c.metric(name.upper() if name == "rmse" else name, f"{cor[name]:.2f}", f"{delta:+.2f} vs raw", delta_color="inverse" if better_low else "normal")
    st.caption(f"Observed heavy-rain cells today: {int(raw.obs_heavy_cells)} · observed max {raw.obs_max:.0f} mm · hits/misses/false alarms — raw {int(raw.hits)}/{int(raw.misses)}/{int(raw.false_alarms)}, corrected {int(cor.hits)}/{int(cor.misses)}/{int(cor.false_alarms)}")

    # ---- district map + table
    st.subheader("District-level product")
    dd = D["dist"][D["dist"].date == sel_date].copy()
    m1, m2 = st.columns([3, 2])
    with m1:
        dl = st.radio("District layer", ["Corrected rain (district mean)", "Corrected rain (district max)", "Raw NWP (district mean)", "Observed (district mean)", "Alert level", "P(> 64.5 mm)"], horizontal=True)
        col = {"Corrected rain (district mean)": "corr_mean", "Corrected rain (district max)": "corr_max", "Raw NWP (district mean)": "raw_mean",
               "Observed (district mean)": "obs_mean", "P(> 64.5 mm)": "p_heavy"}.get(dl)
        fm = folium.Map(location=[22, 80], zoom_start=4.6, tiles="OpenStreetMap")
        if dl == "Alert level":
            lut = dict(zip(dd.district_id, dd.alert))
            folium.GeoJson(D["geo"], style_function=lambda f: dict(fillColor=ALERT_COLORS.get(lut.get(f["properties"]["district_id"], "Green"), "#ccc"), color="#555", weight=0.3, fillOpacity=0.8),
                           tooltip=folium.GeoJsonTooltip(["district", "state"])).add_to(fm)
        else:
            vals = dd[col].dropna()
            if len(vals) and vals.nunique() > 1:
                qs = sorted(set(np.nanpercentile(vals, [0, 20, 40, 60, 80, 100]).tolist()))
                if len(qs) < 3:
                    qs = [float(vals.min()), float(vals.max())]
                folium.Choropleth(geo_data=D["geo"], data=dd, columns=["district_id", col], key_on="feature.properties.district_id",
                                  fill_color="YlGnBu" if "rain" in dl.lower() or "Observed" in dl else "YlOrRd", fill_opacity=0.8, line_opacity=0.2,
                                  nan_fill_color="#eee", legend_name=dl, bins=qs).add_to(fm)
            else:
                st.caption("Not enough variation to shade districts for this layer/date.")
            lut = dd.set_index("district_id")[["district", "state", "raw_mean", "corr_mean", "obs_mean", "p_heavy", "alert"]].to_dict("index")
            for f in D["geo"]["features"]:
                f["properties"].update({k: (round(v, 1) if isinstance(v, float) else v) for k, v in lut.get(f["properties"]["district_id"], {}).items()})
            folium.GeoJson(D["geo"], style_function=lambda f: dict(fillOpacity=0, color="#00000000", weight=0),
                           tooltip=folium.GeoJsonTooltip(["district", "state", "raw_mean", "corr_mean", "obs_mean", "p_heavy", "alert"],
                                                         aliases=["District", "State", "Raw mm", "Corrected mm", "Observed mm", "P(>64.5)", "Alert"])).add_to(fm)
        st_folium(fm, height=520, use_container_width=True, returned_objects=[])
    with m2:
        st.markdown("**Districts flagged (Orange / Red) — sorted by corrected max**")
        top = dd[dd.alert.isin(["Orange", "Red"])].sort_values("corr_max", ascending=False)
        if top.empty:
            top = dd.sort_values("corr_max", ascending=False).head(12)
            st.caption("No Orange/Red today — showing top 12 by corrected max")
        st.dataframe(top[["district", "state", "raw_max", "corr_max", "obs_max", "p_heavy", "p_vheavy", "alert"]].rename(
            columns={"raw_max": "Raw max", "corr_max": "Corr max", "obs_max": "Obs max", "p_heavy": "P>64.5", "p_vheavy": "P>115.5"}).round(2),
            hide_index=True, use_container_width=True, height=480)
        st.caption("Alert: Yellow if P(>64.5) ≥ 0.4 · Orange if P(>115.5) ≥ 0.4 · Red if P(>204.4) ≥ 0.4 (thresholds tunable)")

elif page == "Live forecast (today)":
    st.title("Live operational run")
    live = ROOT / "data" / "live"
    if not (live / "latest.json").exists():
        st.info("No live run yet. Run:  `python scripts/06_predict_today.py`  (fetches today's 00Z GFS, ~1-2 min)")
    else:
        info = json.load(open(live / "latest.json")); ldir = live / info["valid_date"]
        st.markdown(f"**GFS {info['init']} 00Z → Day-{info['lead']} → valid {info['valid_date']}** · produced in **{info['seconds']} s** · "
                    f"<span style='padding:4px 10px;border-radius:6px;background:{REGIME_COLORS[info['regime']]};color:white;font-weight:700'>Regime: {info['regime']}</span> "
                    f"<span style='color:#666'>{' · '.join(f'{k} {v:.2f}' for k, v in info['proba'].items())}</span>", unsafe_allow_html=True)
        g = xr.open_dataset(ldir / "grid.nc").load(); tab = pd.read_csv(ldir / "districts.csv")
        c1, c2, c3 = st.columns(3)
        c1.plotly_chart(grid_map(g["raw"].values, "Raw GFS forecast"), use_container_width=True)
        c2.plotly_chart(grid_map(g["corrected"].values, "Corrected (regime-aware)"), use_container_width=True)
        c3.plotly_chart(grid_map(g["prob"].values[0], "P(> 64.5 mm)", zmax=1, colorscale="YlOrRd", unit="prob"), use_container_width=True)
        st.subheader(f"District alerts — {int((tab.alert == 'Red').sum())} Red · {int((tab.alert == 'Orange').sum())} Orange · {int((tab.alert == 'Yellow').sum())} Yellow")
        m1, m2 = st.columns([3, 2])
        with m1:
            lut = dict(zip(tab.district_id, tab.alert))
            fm = folium.Map(location=[22, 80], zoom_start=4.6, tiles="OpenStreetMap")
            folium.GeoJson(D["geo"], style_function=lambda f: dict(fillColor=ALERT_COLORS.get(lut.get(f["properties"]["district_id"], "Green"), "#ccc"), color="#555", weight=0.3, fillOpacity=0.8),
                           tooltip=folium.GeoJsonTooltip(["district", "state"])).add_to(fm)
            st_folium(fm, height=520, use_container_width=True, returned_objects=[])
        with m2:
            top = tab[tab.alert != "Green"].sort_values("corr_max", ascending=False)
            st.dataframe(top[["district", "state", "raw_max", "corr_max", "p_heavy", "p_vheavy", "p_xheavy", "alert"]].round(2), hide_index=True, use_container_width=True, height=520)

elif page == "Season verification":
    st.title(f"Verification — {dates.min():%b}–{dates.max():%b %Y} test season, Day-1")
    st.caption("Raw GFS vs corrected, both scored against IMD 0.25° observations on identical days. Metrics: xskillscore; FSS after Roberts & Lean (2008).")
    sbs = D["sbs"].copy()
    thr_order = ["all", "heavy (>64.5 mm)", "very heavy (>115.5 mm)", "extremely heavy (>204.4 mm)"]
    sbs["threshold"] = pd.Categorical(sbs["threshold"], thr_order); sbs = sbs.sort_values(["threshold", "metric"])
    corr_col = [c for c in sbs.columns if c.startswith("Corrected")][0]
    show = sbs[["threshold", "metric", "Raw NWP", corr_col, "improvement"]].rename(columns={corr_col: "Corrected", "improvement": "Δ (better = +)"})
    st.dataframe(show.style.format({"Raw NWP": "{:.3f}", "Corrected": "{:.3f}", "Δ (better = +)": "{:+.3f}"})
                 .background_gradient(subset=["Δ (better = +)"], cmap="RdYlGn", vmin=-0.2, vmax=0.2), hide_index=True, use_container_width=True, height=700)
    st.markdown("**Headline metrics at the heavy-rain threshold (> 64.5 mm)**")
    h = show[(show.threshold == "heavy (>64.5 mm)") & (show.metric.isin(["POD", "FAR", "CSI", "ETS", "FSS 3x3", "FSS 5x5"]))]
    fig = go.Figure([go.Bar(name="Raw GFS", x=h.metric, y=h["Raw NWP"], marker_color="#999"),
                     go.Bar(name="Corrected", x=h.metric, y=h["Corrected"], marker_color="#1f77b4")])
    fig.update_layout(barmode="group", height=380, margin=dict(l=10, r=10, t=20, b=10), yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True)
    both = D["mseason"]
    st.markdown("**Shared U-Net vs regime experts (same test days)** — CSI at each threshold")
    piv = both[both.metric == "CSI"].pivot(index="threshold", columns="forecast", values="value").reindex(thr_order[1:]).round(3)
    st.dataframe(piv, use_container_width=True)
    st.markdown(f"Brier scores of the exceedance probabilities — > 64.5: **{head['brier_64.5']:.4f}** · > 115.5: **{head['brier_115.5']:.4f}** · > 204.4: **{head['brier_204.4']:.4f}** (lower is better)")
    st.markdown("**Regime classifier on the test season**")
    ce = D["ceval"]
    st.dataframe(pd.DataFrame(ce["confusion"], index=[f"true {c}" for c in REGIME_COLORS], columns=[f"pred {c}" for c in REGIME_COLORS]), use_container_width=True)
    rep = pd.DataFrame(ce["report"]).T.loc[list(REGIME_COLORS)][["precision", "recall", "f1-score", "support"]].round(2)
    st.dataframe(rep, use_container_width=True)

elif page == "Regime timeline":
    st.title("Regime timeline of the test season")
    reg = D["reg"].copy()
    fig = go.Figure()
    for r, c in REGIME_COLORS.items():
        sub = reg[reg.regime_pred == r]
        fig.add_trace(go.Bar(x=sub.date, y=[1] * len(sub), name=f"predicted {r}", marker_color=c, width=86400000 * 0.9))
    fig.add_trace(go.Scatter(x=reg.date, y=[1.15] * len(reg), mode="markers", name="label (from obs)",
                             marker=dict(color=[REGIME_COLORS[r] for r in reg.regime_label], size=9, symbol="square")))
    fig.update_layout(barmode="stack", height=260, yaxis=dict(visible=False, range=[0, 1.4]), margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Bars: regime predicted by the classifier from the forecast wind & pressure fields. Squares: label derived from IMD observations (Rajeevan index + MSLP low).")
    md = D["mday"]
    st.markdown("**Daily skill (CSI > 64.5 mm): raw vs corrected**")
    fig2 = go.Figure()
    for name, c in [("Raw NWP", "#999"), (f"Corrected ({'regime experts' if head['headline'] == 'experts' else 'shared U-Net'})", "#1f77b4")]:
        s = md[md.forecast == name]; fig2.add_trace(go.Scatter(x=s.date, y=s.CSI, name=name, line=dict(color=c)))
    s = md[md.forecast == "Raw NWP"]
    fig2.add_trace(go.Bar(x=s.date, y=s.obs_heavy_cells / s.obs_heavy_cells.max(), name="observed heavy-rain extent (scaled)", marker_color="rgba(200,50,50,0.25)"))
    fig2.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("**Mean daily skill by predicted regime**")
    t = md.merge(reg[["date", "regime_pred"]], on="date").groupby(["regime_pred", "forecast"])[["rmse", "POD", "FAR", "CSI"]].mean().round(3)
    st.dataframe(t, use_container_width=True)

elif page == "Case gallery":
    st.title("Case gallery — the biggest heavy-rain days of the season")
    md = D["mday"]
    raw = md[md.forecast == "Raw NWP"].set_index("date"); cor = md[md.forecast != "Raw NWP"].groupby("date").first()
    top = raw.sort_values("obs_heavy_cells", ascending=False).head(6)
    for d, r in top.iterrows():
        j = int(np.where(dates == d)[0][0])
        st.markdown(f"### {d:%d %b %Y} — {D['reg'].iloc[j].regime_pred} · {int(r.obs_heavy_cells)} cells > 64.5 mm observed · max {r.obs_max:.0f} mm")
        c1, c2, c3 = st.columns(3)
        c1.plotly_chart(grid_map(pred["raw"].values[j], f"Raw GFS · CSI {r.CSI:.2f}"), use_container_width=True)
        c2.plotly_chart(grid_map(pred[CORR].values[j], f"Corrected · CSI {cor.loc[d].CSI:.2f}"), use_container_width=True)
        c3.plotly_chart(grid_map(pred["obs"].values[j], "Observed (IMD)"), use_container_width=True)

else:
    st.title("About the system")
    st.markdown("""
**Pipeline.** GFS 0.25° (NOAA Open Data, 00Z run) → 24-h totals on the IMD day (fxx 3→27) → IMD 129 × 135 grid → z-score → 6-channel tensor
`[rain, u850, v850, MSLP, elevation, land mask]` (terrain from SRTM 90 m, block-averaged).

**Regime classifier.** CNN on wind & pressure only → Active / Break / Depression / Coastal-Orographic. Labels from observations:
Rajeevan, Gadgil & Bhate (2010) core-zone anomaly index (±1σ, ≥3 days) and closed MSLP lows.

**Correction.** Four regime-specific U-Nets (hard-routed by the classifier), warm-started from one shared U-Net. Output: corrected rain (mm) +
P(> 64.5), P(> 115.5), P(> 204.4 mm). Loss: MSE + λ·penalty on under-predicted heavy cells + BCE on the exceedance channels.

**Verification.** RMSE, ETS, CSI, POD, FAR (xskillscore) and FSS at 1/3/5/9-cell neighbourhoods, raw vs corrected against IMD, on a held-out season.

**Honesty notes.** Prototype trained on two seasons (2023–24), Day-1 only; test season is fully held out. District boundaries: GADM 4.1 (to be replaced by Survey of India).
""")
    st.markdown("**Terrain channel (block-averaged SRTM, m)**")
    st.plotly_chart(grid_map(np.nan_to_num(D["terrain"]["elev_mean"].values), "Mean elevation per 0.25° cell", zmax=4000, colorscale="Earth", unit="m"), use_container_width=True)
