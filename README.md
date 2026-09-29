# SIH26080 — Regime-Aware AI Post-Processing of Monsoon Rainfall Forecasts

Identify the monsoon regime from the forecast itself (Active / Break / Depression / Coastal-Orographic),
route the raw GFS rainfall forecast to a regime-specific U-Net, and deliver district-level corrected rainfall
with calibrated probabilities of exceeding IMD's heavy-rain thresholds (64.5 / 115.5 / 204.4 mm).
Verified against IMD 0.25° observations with RMSE, ETS, CSI, POD, FAR and FSS.

## Run
```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128      # GPU build (optional)

python scripts/01_download.py        # GFS (herbie), IMD (imdlib), SRTM, districts  - resumable
python scripts/02_build_dataset.py   # unified 6-channel tensor + normalisation stats
python scripts/03_labels.py          # regime labels (Rajeevan 2010 index + MSLP lows)
python scripts/04_train.py           # classifier + shared U-Net + 4 regime experts, test-season predictions
python scripts/05_precompute.py      # district tables, metrics, everything the dashboard reads
python scripts/06_predict_today.py   # operational run on today's 00Z GFS
streamlit run dashboard/app.py
```

## Layout
```
configs/config.yaml          years, leads, thresholds, regime parameters
src/rainfall/grid.py         master grid = IMD 0.25° (129 lat × 135 lon)
src/rainfall/data/           gfs.py imd.py dem.py districts.py build_dataset.py
src/rainfall/regimes/        labels.py (Rajeevan index, MSLP lows)  classifier.py (CNN gatekeeper)
src/rainfall/correction/     unet.py (U-Net, asymmetric loss, 4 experts)
src/rainfall/aggregate/      districts.py (area-weighted mean / max, alert colours)
src/rainfall/verify/         metrics.py (xskillscore + FSS)
dashboard/app.py             Streamlit + Folium
docs/                        flowcharts (SVG) + generator
```
Time convention: IMD's daily rainfall for date D is the 24 h ending 03 UTC on D, so Day-1 from the 00Z run
uses forecast hours 3→27 (`gfs.py`).
