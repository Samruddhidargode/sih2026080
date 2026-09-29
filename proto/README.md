# Regime-aware rainfall correction: prototype v2

A simple, explainable prototype that matches the SIH design slides:
**identify the regime (2 layers) → correct with 4 experts → soft blend + calibrate → district alerts → verify.**
It reuses the existing processed data in `data/processed/` (GFS 0.25° Day-1 + IMD 0.25° truth, JJAS 2023–2025);
the original prototype in `src/`, `scripts/` and `dashboard/` is untouched.

## Run
```bash
.venv/Scripts/python.exe proto/run_pipeline.py      # ~10-30 min (U-Net trains 3x on GPU)
.venv/Scripts/python.exe -m streamlit run proto/app.py
```

## What each file does
| File | Role |
|---|---|
| `common.py` | paths, thresholds, years, labels |
| `features.py` | physics: terrain/coast, **Layer 2** local type (orographic / coastal / plains from forecast winds), day indices, LightGBM features |
| `regimes.py` | **Layer 1** day regime (LPS / Active / Break / Normal) as probabilities, LightGBM on forecast indices |
| `experts.py` | **E0** box quantile mapping (single-method baseline) · **E1** regime-aware QM · **E2** LightGBM · **E3** U-Net (existing code) |
| `blend.py` | soft blending (weights per situation, learned out-of-fold, mixed by probability) + isotonic / logistic calibration |
| `verify.py` | RMSE, POD, FAR, CSI, ETS, FSS, Brier/BSS, block bootstrap |
| `run_pipeline.py` | leave-one-year-out on 2023/2024 → blend fitted out-of-fold → final models → scored once on 2025 |
| `app.py` | 5-step story app: problem → regime → correction → alerts → proof |

## Honesty rules built in
- Split by **whole years**; 2025 is used once, for the final scores.
- Blend weights, calibration and probability cut-offs are fitted only on **out-of-fold** predictions.
- The regime is predicted **from the forecast**; observed labels are used only for training and scoring.
- Machine-specific: scikit-learn is blocked by Windows Application Control here, so LightGBM's native API
  and NumPy implementations (isotonic, metrics) are used instead.
