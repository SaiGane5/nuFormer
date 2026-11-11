"""Tabular feature builder — deliberately lossy (coarse categories, 90-day window, noisy declared fields)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .catalog import COARSE
from .config import GenConfig

COARSE_COLS = ["grocery", "food", "transport", "shopping", "health", "bills", "transfers", "other"]


def build_tabular(p: pd.DataFrame, tx: pd.DataFrame, score_dates: np.ndarray, cfg: GenConfig,
                  rng: np.random.Generator) -> pd.DataFrame:
    n = len(p)
    sd = pd.Series(score_dates, index=p.user_id)
    t = tx[tx.date >= (sd.reindex(tx.user_id).to_numpy() - np.timedelta64(90, "D"))]
    g = t.groupby("user_id")
    f = pd.DataFrame({"user_id": p.user_id})
    f["age"] = p.age
    f["tenure_months"] = p.tenure_months
    declared = p.income_monthly * np.exp(rng.normal(0, 0.35, n))
    f["declared_income"] = np.where(rng.random(n) < 0.15, -1.0, declared.round(-2))
    f["credit_limit"] = (np.clip(p.income_monthly * np.exp(0.4 * p.z_saver - 0.5 * p.z_stress + rng.normal(0, 0.3, n)),
                                 200, 80_000)).round(-2)
    f["app_sessions_30d"] = rng.poisson(12 * np.exp(0.6 * p.z_digital))
    f["has_kids_declared"] = np.where(rng.random(n) < 0.8, p.has_kids, ~p.has_kids).astype(int)
    f["region"] = p.region
    idx = p.user_id
    f["n_tx_90d"] = g.size().reindex(idx, fill_value=0).to_numpy()
    f["inflow_90d"] = t[t.amount > 0].groupby("user_id").amount.sum().reindex(idx, fill_value=0).round(2).to_numpy()
    f["outflow_90d"] = -t[t.amount < 0].groupby("user_id").amount.sum().reindex(idx, fill_value=0).round(2).to_numpy()
    out = t[t.amount < 0].assign(coarse=lambda d: d.category.map(lambda c: COARSE.get(c, "other")))
    piv = out.pivot_table(index="user_id", columns="coarse", values="amount", aggfunc="sum").reindex(idx).fillna(0)
    for c in COARSE_COLS:
        f[f"spend_{c}_90d"] = (-piv[c]).round(2).to_numpy() if c in piv else 0.0
    f["overdraft_flag_90d"] = t[t.category == "FEE"].groupby("user_id").size().reindex(idx, fill_value=0).gt(0).astype(int).to_numpy()
    f["channel"] = rng.choice(cfg.channels, n, p=[0.5, 0.2, 0.3])
    return f
