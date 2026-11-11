"""Latent persona sampling (Gaussian copula over traits) and life events."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GenConfig

TRAITS = ["inc", "saver", "travel", "stress", "digital", "disc", "age"]
# target correlation between latent gaussians (order = TRAITS)
_R = np.array([
    #  inc  saver travel stress digit  disc   age
    [1.00, 0.40, 0.40, -0.40, 0.10, 0.10, 0.30],
    [0.40, 1.00, 0.05, -0.45, 0.00, -0.30, 0.20],
    [0.40, 0.05, 1.00, -0.20, 0.20, 0.30, 0.00],
    [-0.40, -0.45, -0.20, 1.00, 0.00, 0.20, -0.10],
    [0.10, 0.00, 0.20, 0.00, 1.00, 0.20, -0.40],
    [0.10, -0.30, 0.30, 0.20, 0.20, 1.00, -0.20],
    [0.30, 0.20, 0.00, -0.10, -0.40, -0.20, 1.00],
])

EVENT_TYPES = ("NONE", "RAISE", "STRESS", "NEW_CAR", "BABY")


def sample_personas(cfg: GenConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.n_users
    L = np.linalg.cholesky(_R)                      # raises if not PSD -> design-time guard
    z = rng.standard_normal((n, len(TRAITS))) @ L.T
    df = pd.DataFrame(z, columns=[f"z_{t}" for t in TRAITS])
    df.insert(0, "user_id", np.arange(n, dtype=np.int64))
    df["income_monthly"] = cfg.median_monthly_income * np.exp(0.65 * df.z_inc)
    df["age"] = np.clip(np.round(38 + 11 * df.z_age), 18, 75).astype(int)
    df["tenure_months"] = np.clip(rng.gamma(2.0, 18.0, n), 1, 140).round().astype(int)
    df["has_kids"] = rng.random(n) < 1 / (1 + np.exp(-(df.age - 33) / 6)) * 0.7
    df["has_car"] = rng.random(n) < 1 / (1 + np.exp(-(0.9 * df.z_inc + 0.2)))
    df["region"] = rng.choice(cfg.regions, n, p=[0.45, 0.15, 0.2, 0.1, 0.1])
    df["pays_rent"] = rng.random(n) < 0.55
    df["salary_day"] = rng.choice([1, 5, 10, 15, 30], n, p=[0.15, 0.45, 0.1, 0.15, 0.15])
    df["n_subs"] = rng.binomial(5, 1 / (1 + np.exp(-(0.8 * df.z_digital + 0.4 * df.z_inc - 0.5))))
    return df


def sample_life_events(p: pd.DataFrame, cfg: GenConfig, rng: np.random.Generator) -> pd.DataFrame:
    """At most one life event per user; `event_day` = days before score date (1 = yesterday)."""
    n = len(p)
    w = np.stack([
        np.zeros(n),                                            # placeholder for NONE (handled below)
        np.exp(0.6 * p.z_inc + 0.3 * p.z_saver),                # RAISE
        np.exp(1.0 * p.z_stress),                               # STRESS
        np.exp(0.5 * p.z_inc) * (~p.has_car + 0.3),             # NEW_CAR
        np.exp(-((p.age - 31) / 6) ** 2) * 2.5,                 # BABY
    ], 1)
    w = w / w.sum(1, keepdims=True)
    has = rng.random(n) < cfg.p_life_event
    choice = np.array([rng.choice(5, p=wi) for wi in w])
    ev = np.where(has, choice, 0)
    # recency: skewed toward the recent past so "≤30 days" happens often (~35%)
    day = np.clip(np.round(rng.gamma(1.3, 30.0, n)), 1, cfg.history_days - 5).astype(int)
    return pd.DataFrame({"user_id": p.user_id, "event": np.array(EVENT_TYPES)[ev], "event_day": np.where(ev > 0, day, -1)})
