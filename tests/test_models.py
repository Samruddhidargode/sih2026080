import numpy as np, torch
from rainfall.correction.unet import make_expert, asymmetric_loss, predict
from rainfall.regimes.labels import CLASSES, _runs, active_break
import pandas as pd


def test_experts_shapes_and_loss():
    x = torch.randn(2, 6, 129, 135); y = torch.rand(2, 129, 135) * 100; y[:, :30] = float("nan")
    for r in CLASSES:
        out = make_expert(r, base=8)(x)
        assert out.shape == (2, 4, 129, 135)
        loss, parts = asymmetric_loss(out, y)
        assert torch.isfinite(loss)


def test_probabilities_monotone():
    rain, prob = predict(make_expert("Active", base=8), np.random.randn(3, 6, 129, 135).astype("float32"))
    assert (rain >= 0).all() and (np.diff(prob, axis=1) <= 1e-6).all()


def test_runs_and_active_break():
    assert _runs(np.array([1, 1, 0, 1, 1, 1, 0], bool), 3).tolist() == [False, False, False, True, True, True, False]
    idx = pd.date_range("2024-07-01", periods=8)
    lab = active_break(pd.Series([1.5, 1.5, 1.5, 0, -1.5, -1.5, 0, 2], index=idx))
    assert lab.tolist()[:3] == ["Active"] * 3 and lab.iloc[4] is None and lab.iloc[7] is None
