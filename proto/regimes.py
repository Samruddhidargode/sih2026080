"""
Layer 1: large-scale regime of the day, predicted FROM THE FORECAST as probabilities.
  - 3-class LightGBM:  Active / Break / Normal
  - binary LightGBM:   monsoon low / depression (LPS) present
Combined into 4 day situations: LPS, Active, Break, Normal (LPS overrides).
Layer 2 (per box) comes from features.LocalLayer; the 12 combined situation
probabilities are  P(day situation) x P(local type).
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import DAY_SIT

G1 = ["Active", "Break", "Normal"]
PARAMS = dict(learning_rate=0.05, num_leaves=7, min_data_in_leaf=8, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7)


class DayRegimeClassifier:
    def fit(self, df: pd.DataFrame, lab: pd.DataFrame):
        g1 = lab["active_break"].fillna("Normal").map({c: i for i, c in enumerate(G1)}).values
        lps = lab["depression"].astype(int).values
        freq = np.bincount(g1, minlength=3) + 1
        w = (1.0 / np.sqrt(freq))[g1]; w = w / w.mean()          # soften class imbalance
        self.m1 = lgb.train({**PARAMS, "objective": "multiclass", "num_class": 3},
                            lgb.Dataset(df.values, g1, weight=w), 150)
        self.m2 = lgb.train({**PARAMS, "objective": "binary"}, lgb.Dataset(df.values, lps), 150)
        self.cols = list(df.columns)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        p1 = self.m1.predict(df[self.cols].values); pl = self.m2.predict(df[self.cols].values)
        out = pd.DataFrame({"LPS": pl, "Active": (1 - pl) * p1[:, 0], "Break": (1 - pl) * p1[:, 1],
                            "Normal": (1 - pl) * p1[:, 2]}, index=df.index)
        return out[DAY_SIT]


def situation_probs(pday: pd.DataFrame, P_local: np.ndarray) -> np.ndarray:
    """(12, n, H, W): P(day situation) x P(local type)."""
    pd_ = pday[DAY_SIT].values.T.astype("float32")               # (4, n)
    return (pd_[:, None, :, None, None] * P_local[None]).reshape(12, *P_local.shape[1:])
