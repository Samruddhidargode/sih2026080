"""
Regime-aware rainfall correction: a 5-step story of the problem and the solution.
Reads only precomputed files from proto/output (run proto/run_pipeline.py first).

    .venv/Scripts/python.exe -m streamlit run proto/app.py
"""
import json
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xarray as xr
from streamlit_folium import st_folium

OUT = Path(__file__).resolve().parent / "output"
st.set_page_config(page_title="Regime-Aware Rainfall Correction", layout="wide", page_icon="🌧️")

RAIN = [[0, "#ffffff"], [0.02, "#d4ecf7"], [0.08, "#7cc4e8"], [0.16, "#2b8cbe"], [0.28, "#0b5394"],
        [0.4, "#31a354"], [0.55, "#ffd92f"], [0.7, "#fd8d3c"], [0.85, "#e31a1c"], [1, "#67000d"]]
SIT_COL = {"LPS": "#6A3FA0", "Active": "#1F5FAD", "Break": "#E08E0B", "Normal": "#607D8B"}
LOC_COL = {"Orographic": "#8D6E63", "Coastal": "#0F7C7C", "Plains": "#E6E0C8"}
ALERT = {"Green": "#7fbf7f", "Yellow": "#f5d442", "Orange": "#f28c28", "Red": "#d62728"}
EXP_COL = ["#90A4AE", "#0F7C7C", "#E08E0B", "#6A3FA0"]
MAIN = ["Raw GFS", "E0 Quantile map", "Regime-aware blend"]


@st.cache_resource
def load():
    return dict(f=xr.open_dataset(OUT / "fields_2025.nc"), meta=json.load(open(OUT / "meta.json")),
                day=pd.read_csv(OUT / "day_table.csv", parse_dates=["date"]),
                season=pd.read_csv(OUT / "scores_season.csv"), strat=pd.read_csv(OUT / "scores_by_regime.csv"),
                diag=pd.read_csv(OUT / "diag_problem.csv"),
                dist=pd.read_csv(OUT / "district_daily.csv", parse_dates=["date"]),
                geo=json.load(open(OUT / "districts.geojson")))


if not (OUT / "meta.json").exists():
    st.error("No outputs yet. Run:  `.venv/Scripts/python.exe proto/run_pipeline.py`"); st.stop()
D = load(); F = D["f"]; meta = D["meta"]; day = D["day"]
dom = F["domain"].values.astype(bool); LAT, LON = F["lat"].values, F["lon"].values

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🌧️ Regime-aware rainfall correction")
st.sidebar.caption("SIH problem: identify the weather regime, then correct the raw NWP rainfall forecast.")
dates = list(day["date"])
default = int(np.argmax(day["obs_heavy_cells"].values))
sel = st.sidebar.selectbox("Forecast day (2025 test season, Day-1)", dates, index=default, format_func=lambda d: d.strftime("%d %b %Y"))
i = dates.index(sel); r = day.iloc[i]
st.sidebar.markdown("---")
st.sidebar.markdown("**How to read this demo**\n\n① The problem\n\n② Identify the regime\n\n③ Correct the forecast\n\n④ Heavy-rain alerts\n\n⑤ Does it work?")
st.sidebar.markdown("---")
st.sidebar.caption(f"Trained on JJAS {meta['train_years'][0]}–{meta['train_years'][-1]} · tested once on the unseen {meta['test_year']} season · "
                   "GFS 0.25° (NOAA open data, stand-in for IMD GFS) vs IMD 0.25° observed rainfall.")


def rain_map(a, title, zmax=150, scale=RAIN, unit="mm", h=380):
    a = np.where(dom, a, np.nan)
    fig = go.Figure(go.Heatmap(z=a, x=LON, y=LAT, zmin=0, zmax=zmax, colorscale=scale, colorbar=dict(title=unit, thickness=10),
                               hovertemplate="%{y:.2f}N %{x:.2f}E<br>%{z:.1f} " + unit + "<extra></extra>"))
    fig.add_trace(go.Contour(z=F["land"].values, x=LON, y=LAT, contours=dict(start=0.5, end=0.5, size=1, coloring="none"),
                             line=dict(color="#333", width=0.6), showscale=False, hoverinfo="skip"))
    fig.update_layout(title=dict(text=title, x=0.5, font=dict(size=14)), height=h, margin=dict(l=5, r=5, t=36, b=5),
                      xaxis=dict(range=[67, 98], showgrid=False, visible=False), yaxis=dict(range=[7, 37], scaleanchor="x", showgrid=False, visible=False))
    return fig


def cat_map(codes, cmap, names, title, h=380):
    k = len(names); z = np.where(dom, codes, np.nan)
    scale = []
    for j, n in enumerate(names):
        scale += [[j / k, cmap[n]], [(j + 1) / k, cmap[n]]]
    fig = go.Figure(go.Heatmap(z=z, x=LON, y=LAT, zmin=-0.5, zmax=k - 0.5, colorscale=scale, showscale=False,
                               customdata=np.array(names, dtype=object)[np.nan_to_num(z).astype(int)],
                               hovertemplate="%{customdata}<extra></extra>"))
    fig.add_trace(go.Contour(z=F["land"].values, x=LON, y=LAT, contours=dict(start=0.5, end=0.5, size=1, coloring="none"),
                             line=dict(color="#333", width=0.6), showscale=False, hoverinfo="skip"))
    fig.update_layout(title=dict(text=title, x=0.5, font=dict(size=14)), height=h, margin=dict(l=5, r=5, t=36, b=5),
                      xaxis=dict(range=[67, 98], visible=False), yaxis=dict(range=[7, 37], scaleanchor="x", visible=False))
    return fig


def legend(cmap):
    return " ".join(f"<span style='display:inline-block;width:12px;height:12px;background:{c};border:1px solid #999;margin:0 4px 0 12px'></span>{n}"
                    for n, c in cmap.items())


def badge(text, colour):
    return f"<span style='padding:5px 12px;border-radius:14px;background:{colour};color:white;font-weight:700'>{text}</span>"


st.markdown(f"## {sel:%A, %d %B %Y}  ·  Day-1 forecast   "
            f"<span style='font-size:15px;color:#666'>observed: {r.obs_heavy_cells} grid boxes with heavy rain (> 64.5 mm), max {r.obs_max:.0f} mm</span>",
            unsafe_allow_html=True)
t1, t2, t3, t4, t5 = st.tabs(["① The problem", "② Identify the regime", "③ Correct the forecast", "④ Heavy-rain alerts", "⑤ Does it work?"])
raw, obs = F["raw"].isel(date=i).values, F["obs"].isel(date=i).values

# ------------------------------------------------------------------ 1. problem
with t1:
    st.markdown("### Weather models miss heavy rain, and *how* they miss it depends on the type of weather")
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(rain_map(raw, "What the model forecast (raw GFS)"), use_container_width=True)
    c2.plotly_chart(rain_map(obs, "What actually fell (IMD observed)"), use_container_width=True)
    err = np.where(dom, raw - obs, np.nan)
    c3.plotly_chart(rain_map(err, "Model error (forecast − observed)", zmax=None, scale="RdBu", unit="mm").update_traces(zmin=-80, zmax=80, selector=dict(type="heatmap")), use_container_width=True)
    heavy_raw = int(np.nansum((raw > 64.5) & dom)); heavy_obs = int(np.nansum(obs > 64.5)); hits = int(np.nansum((raw > 64.5) & (obs > 64.5)))
    st.info(f"On this day heavy rain fell on **{heavy_obs}** grid boxes. The raw model forecast heavy rain on **{heavy_raw}**, and only **{hits}** of those were in the right place.")

    dg = D["diag"].copy()
    st.markdown("#### The model's error is not the same everywhere (JJAS 2023–2024)")
    ca, cb = st.columns(2)
    fig = go.Figure(go.Bar(x=dg["group"], y=dg["heavy_freq_bias"], marker_color=["#1F5FAD"] * 4 + ["#8D6E63", "#0F7C7C", "#BCAA7A"],
                           text=[f"{v:.2f}" for v in dg["heavy_freq_bias"]], textposition="outside"))
    fig.add_hline(y=1, line_dash="dash", line_color="#2E7D32", annotation_text="perfect = 1.0")
    fig.update_layout(title="How often the model forecasts heavy rain, vs how often it happens", height=380,
                      yaxis=dict(title="forecast ÷ observed heavy-rain days", range=[0, 1.25]), margin=dict(l=10, r=10, t=40, b=10))
    ca.plotly_chart(fig, use_container_width=True)
    fig = go.Figure(go.Bar(x=dg["group"], y=dg["total_ratio"], marker_color=["#1F5FAD"] * 4 + ["#8D6E63", "#0F7C7C", "#BCAA7A"],
                           text=[f"{v:.2f}" for v in dg["total_ratio"]], textposition="outside"))
    fig.add_hline(y=1, line_dash="dash", line_color="#2E7D32", annotation_text="perfect = 1.0")
    fig.update_layout(title="Total rain forecast ÷ total rain observed", height=380, yaxis=dict(range=[0, 1.3]),
                      margin=dict(l=10, r=10, t=40, b=10))
    cb.plotly_chart(fig, use_container_width=True)
    cst = dg.set_index("group")
    st.markdown(f"**Reading it:** on **coastal** boxes the model forecasts only **{cst.loc['Local: Coastal', 'heavy_freq_bias']:.0%}** as many heavy-rain cases "
                f"as really occur and **{cst.loc['Local: Coastal', 'total_ratio']:.0%}** of the total rain, while on the **plains** it is "
                f"{'too wet' if cst.loc['Local: Plains', 'total_ratio'] > 1 else 'closer to reality'} ({cst.loc['Local: Plains', 'total_ratio']:.0%}). "
                "**One correction applied everywhere cannot fix errors that point in different directions.** That is why we identify the regime first.")

# ------------------------------------------------------------------ 2. regimes
with t2:
    st.markdown("### Step 1: identify the regime at two levels, from the forecast itself")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("**Layer 1: what kind of day is it? (all India)**")
        probs = {s: r[f"p_{s}"] for s in SIT_COL}
        fig = go.Figure(go.Bar(y=list(probs), x=list(probs.values()), orientation="h", marker_color=list(SIT_COL.values()),
                               text=[f"{v:.0%}" for v in probs.values()], textposition="outside"))
        fig.update_layout(height=260, xaxis=dict(range=[0, 1.15], tickformat=".0%"), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"Predicted: {badge(r.pred, SIT_COL[r.pred])} &nbsp; Observed afterwards: {badge(r.truth, SIT_COL[r.truth])}", unsafe_allow_html=True)
        st.caption("LPS = monsoon low / depression. Probabilities come from a LightGBM classifier reading the forecast's "
                   "monsoon trough, low-level winds and pressure lows, not from observations.")
        st.markdown(f"<br>Test-season accuracy of Layer 1: **{meta['classifier']['test_accuracy']:.0%}** (4 classes). "
                    "Because we blend by probability, a wrong guess shifts weights gently instead of switching models.", unsafe_allow_html=True)
    with c2:
        st.markdown("**Layer 2: what kind of rain does each place get today?**")
        pl = F["p_local"].isel(date=i).values
        st.plotly_chart(cat_map(np.argmax(pl, 0), LOC_COL, list(LOC_COL), "Local rain type per grid box", h=420), use_container_width=True)
        st.markdown(legend(LOC_COL), unsafe_allow_html=True)
        st.caption("Orographic = moist wind forced up a mountain slope · Coastal = wind blowing from the sea onto land · computed from forecast winds and terrain.")
    st.markdown("**Season timeline: regime predicted from the forecast vs observed**")
    fig = go.Figure()
    for s, c in SIT_COL.items():
        sub = day[day.pred == s]
        fig.add_trace(go.Bar(x=sub.date, y=[1] * len(sub), marker_color=c, name=f"predicted {s}", width=86400000 * 0.9))
    fig.add_trace(go.Scatter(x=day.date, y=[1.18] * len(day), mode="markers", name="observed", marker=dict(color=[SIT_COL[s] for s in day.truth], symbol="square", size=8)))
    fig.update_layout(barmode="stack", height=200, yaxis=dict(visible=False, range=[0, 1.35]), margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ 3. correction
with t3:
    st.markdown("### Step 2 + 3: four experts correct the forecast, blended by regime")
    c = st.columns(4)
    c[0].plotly_chart(rain_map(raw, "Raw GFS"), use_container_width=True)
    c[1].plotly_chart(rain_map(F["amt_e0"].isel(date=i).values, "Single method (quantile map)"), use_container_width=True)
    c[2].plotly_chart(rain_map(F["amt_blend"].isel(date=i).values, "Regime-aware blend (ours)"), use_container_width=True)
    c[3].plotly_chart(rain_map(obs, "Observed (IMD)"), use_container_width=True)
    m = st.columns(3)
    for col, (nm, key) in zip(m, [("Raw GFS", "csi_raw"), ("Single method", "csi_e0"), ("Regime-aware blend", "csi_blend")]):
        col.metric(f"{nm}: heavy-rain CSI today", f"{r[key]:.2f}")
    st.caption("CSI = hits ÷ (hits + misses + false alarms) for rain > 64.5 mm. 1 = perfect, 0 = no skill.")

    with st.expander("Inside the blend: the four experts and who was trusted where", expanded=True):
        c = st.columns(4)
        labels = ["E0 · quantile mapping", "E1 · regime-wise QM", "E2 · LightGBM (physics features)", "E3 · U-Net CNN (moves rain)"]
        for k in range(4):
            c[k].plotly_chart(rain_map(F[f"amt_e{k}"].isel(date=i).values, labels[k], h=300), use_container_width=True)
        w = F["w_expert"].isel(date=i).values
        cc1, cc2 = st.columns([3, 2])
        cmap = dict(zip(["E0", "E1", "E2", "E3"], EXP_COL))
        cc1.plotly_chart(cat_map(np.argmax(w, 0), cmap, list(cmap), "Expert with the largest weight in each box", h=360), use_container_width=True)
        cc1.markdown(legend(cmap), unsafe_allow_html=True)
        mw = [float(np.nanmean(np.where(dom, w[k], np.nan))) for k in range(4)]
        fig = go.Figure(go.Bar(x=["E0", "E1", "E2", "E3"], y=mw, marker_color=EXP_COL, text=[f"{v:.0%}" for v in mw], textposition="outside"))
        fig.update_layout(title="Average weight today", height=300, yaxis=dict(tickformat=".0%", range=[0, max(mw) * 1.3]), margin=dict(l=10, r=10, t=40, b=10))
        cc2.plotly_chart(fig, use_container_width=True)
        cc2.caption("Weights were learned per situation (day regime × local type) on years the experts never trained on, "
                    "then mixed box-by-box using today's regime probabilities.")

# ------------------------------------------------------------------ 4. alerts
with t4:
    st.markdown("### From grid boxes to district warnings")
    dd = D["dist"][D["dist"].date == sel].copy()
    c1, c2 = st.columns(2)
    c1.plotly_chart(rain_map(F["prob_blend"].isel(date=i, thr=0).values, "Chance of heavy rain (> 64.5 mm)", zmax=1,
                             scale="YlOrRd", unit="prob"), use_container_width=True)
    c2.plotly_chart(rain_map(F["prob_blend"].isel(date=i, thr=1).values, "Chance of very heavy rain (> 115.5 mm)", zmax=1,
                             scale="YlOrRd", unit="prob"), use_container_width=True)

    def dmap(col):
        lut = dict(zip(dd.district_id, dd[col]))
        fm = folium.Map(location=[22.5, 80], zoom_start=4.4, tiles="CartoDB positron")
        folium.GeoJson(D["geo"], style_function=lambda f: dict(fillColor=ALERT.get(lut.get(f["properties"]["district_id"], "Green"), "#ccc"),
                                                               color="#666", weight=0.3, fillOpacity=0.85),
                       tooltip=folium.GeoJsonTooltip(["district", "state"])).add_to(fm)
        return fm
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("**Our forecast alert (from probabilities)**")
        st_folium(dmap("alert"), height=430, use_container_width=True, returned_objects=[], key=f"fa{i}")
    with a2:
        st.markdown("**What actually happened (observed district max)**")
        st_folium(dmap("alert_obs"), height=430, use_container_width=True, returned_objects=[], key=f"ob{i}")
    ps = meta["pstar"]["Regime-aware blend"]
    st.markdown(legend(ALERT) + f" &nbsp; <span style='color:#666'>(Yellow: P(>64.5) ≥ {ps['64.5']:.0%} · Orange: P(>115.5) ≥ {ps['115.5']:.0%} · "
                f"Red: P(>204.4) ≥ {ps['115.5']:.0%}; cut-offs tuned on 2023–24 out-of-fold data)</span>", unsafe_allow_html=True)
    hit = int(((dd.alert != "Green") & (dd.alert_obs != "Green")).sum()); obs_n = int((dd.alert_obs != "Green").sum())
    raw_hit = int(((dd.alert_raw != "Green") & (dd.alert_obs != "Green")).sum()); fa = int(((dd.alert != "Green") & (dd.alert_obs == "Green")).sum())
    k1, k2, k3 = st.columns(3)
    k1.metric("Districts that really got heavy rain", obs_n)
    k2.metric("…warned by our system", hit, f"{hit - raw_hit:+d} vs raw model")
    k3.metric("Warnings where heavy rain did not occur", fa)
    top = dd[dd.alert.isin(["Orange", "Red"])].sort_values("blend_max", ascending=False)
    if top.empty:
        top = dd.sort_values("p64", ascending=False).head(12)
    st.dataframe(top[["district", "state", "raw_max", "blend_max", "obs_max", "p64", "p115", "alert", "alert_obs"]].rename(columns={
        "raw_max": "Raw max mm", "blend_max": "Corrected max mm", "obs_max": "Observed max mm", "p64": "P(>64.5)", "p115": "P(>115.5)",
        "alert": "Our alert", "alert_obs": "Observed"}), hide_index=True, use_container_width=True, height=300)

# ------------------------------------------------------------------ 5. proof
with t5:
    st.markdown(f"### Scored once on the unseen {meta['test_year']} monsoon (122 days), against IMD observations")
    s = D["season"].set_index("method")
    show = s[["RMSE", "Bias", "POD@64.5", "FAR@64.5", "CSI@64.5", "ETS@64.5", "FSS3@64.5", "FSS9@64.5", "BSS@64.5", "CSI@115.5"]]
    st.dataframe(show.style.format("{:.3f}", na_rep="—").highlight_max(subset=["POD@64.5", "CSI@64.5", "ETS@64.5", "FSS3@64.5", "FSS9@64.5", "BSS@64.5", "CSI@115.5"], color="#c8e6c9")
                 .highlight_min(subset=["RMSE", "FAR@64.5"], color="#c8e6c9"), use_container_width=True)
    st.caption("@64.5 = heavy rain, @115.5 = very heavy. POD = share of heavy-rain boxes caught · FAR = share of heavy-rain forecasts that were wrong · "
               "CSI / ETS = overall hit skill · FSS9 = right rain within ~110 km · BSS = probability skill vs climatology (> 0 is useful). "
               "Bias = total forecast ÷ total observed rain.")
    c1, c2 = st.columns(2)
    mets = ["POD@64.5", "CSI@64.5", "ETS@64.5", "FSS9@64.5"]
    fig = go.Figure([go.Bar(name=m_, x=mets, y=[s.loc[m_, k] for k in mets], marker_color=c_)
                     for m_, c_ in zip(MAIN, ["#9E9E9E", "#E08E0B", "#6A3FA0"])])
    fig.update_layout(barmode="group", title="Heavy-rain skill (higher is better)", height=380, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    c1.plotly_chart(fig, use_container_width=True)
    stt = D["strat"]
    stt = stt[stt.method.isin(MAIN)]
    fig = go.Figure([go.Bar(name=m_, x=stt[stt.method == m_].group, y=stt[stt.method == m_]["CSI@64.5"], marker_color=c_)
                     for m_, c_ in zip(MAIN, ["#9E9E9E", "#E08E0B", "#6A3FA0"])])
    fig.update_layout(barmode="group", title="Heavy-rain CSI in each weather type", height=380, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    c2.plotly_chart(fig, use_container_width=True)
    dist = D["dist"]; o_ = dist.alert_obs != "Green"
    def dscore(f):
        h, m_, fa = int((f & o_).sum()), int((~f & o_).sum()), int((f & ~o_).sum())
        return dict(POD=h / (h + m_), FAR=fa / max(h + fa, 1), CSI=h / (h + m_ + fa))
    dr, db = dscore(dist.alert_raw != "Green"), dscore(dist.alert != "Green")
    st.markdown("#### District warnings over the whole season (district had heavy rain somewhere vs was warned)")
    k = st.columns(3)
    for col, key, better_low in zip(k, ["POD", "FAR", "CSI"], [False, True, False]):
        col.metric(f"District {key}: ours (raw GFS)", f"{db[key]:.2f}", f"{db[key] - dr[key]:+.2f} vs raw {dr[key]:.2f}",
                   delta_color="inverse" if better_low else "normal")
    b = meta["bootstrap"]
    def ci(d): return f"{d['diff']:+.3f} (95% range {d['lo']:+.3f} to {d['hi']:+.3f})"
    st.markdown(f"**Is the gain real?** Heavy-rain CSI difference, 5-day block bootstrap: blend − raw = {ci(b['blend_vs_raw'])} · "
                f"blend − single method = {ci(b['blend_vs_E0'])}. A range that stays above 0 means the improvement is not luck.")
    st.markdown("**Honest notes.** Prototype trained on only two monsoon seasons (2023–24), Day-1 lead only. NOAA GFS stands in for IMD GFS T1534; "
                "the operational system retrains on IMD / NCMRWF archives. Probabilities are calibrated on out-of-fold years; the 2025 season was used once.")
