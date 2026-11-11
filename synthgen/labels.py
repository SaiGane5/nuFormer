"""Label oracle (LLD 2.2): true per-product propensities, offer policy and Bernoulli labels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import GenConfig

DISCRETIONARY = {"RESTAURANT", "DELIVERY", "SHOPPING", "TRAVEL_ABROAD"}


def _sig(x):
    return 1 / (1 + np.exp(-x))


def _zs(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-9)


def true_logits(p: pd.DataFrame, ev: pd.DataFrame, tx: pd.DataFrame, cfg: GenConfig,
                rng: np.random.Generator) -> np.ndarray:
    out = tx[tx.amount < 0]
    tot = out.groupby("user_id").amount.sum().reindex(p.user_id, fill_value=0).abs()
    disc = out[out.category.isin(DISCRETIONARY)].groupby("user_id").amount.sum().reindex(p.user_id, fill_value=0).abs()
    disc_share = _zs((disc / (tot + 1)).to_numpy())
    e, d = ev.event.to_numpy(), ev.event_day.to_numpy()
    recent = lambda t, w: ((e == t) & (d <= w)).astype(float)
    family = _zs(p.has_kids.astype(float) + 0.5 * (e == "BABY"))
    age = _zs(p.age)
    L = np.stack([
        1.6 * p.z_travel + 0.6 * p.z_inc - 0.5 * p.z_stress,                                      # TRAVEL_CARD
        1.4 * p.z_stress + 1.6 * recent("STRESS", 30) - 0.6 * p.z_saver,                           # PERSONAL_LOAN
        1.5 * p.z_saver + 0.7 * p.z_inc + 1.2 * recent("RAISE", 30) - 0.8 * p.z_stress,            # INVESTMENTS
        1.0 * family + 1.6 * (recent("NEW_CAR", 45) + recent("BABY", 45)) + 0.3 * age,             # INSURANCE
        1.2 * disc_share + 0.5 * p.z_digital,                                                      # CASHBACK_PLUS
    ], 1)
    L = L + rng.normal(0, cfg.label_noise_sd, L.shape)
    # calibrate intercepts so each product has the target base rate
    for k in range(L.shape[1]):
        lo, hi = -10.0, 10.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if _sig(L[:, k] + mid).mean() < cfg.target_base_rate else (lo, mid)
        L[:, k] += (lo + hi) / 2
    return L


def assign_offers_and_labels(L: np.ndarray, cfg: GenConfig, rng: np.random.Generator):
    n, K = L.shape
    targeted = rng.random(n) < cfg.targeted_offer_share
    offer = np.where(targeted, np.argmax(L + rng.normal(0, 1.0, L.shape), 1), rng.integers(0, K, n))
    p_offer = _sig(L[np.arange(n), offer])
    y = (rng.random(n) < p_offer).astype(np.int8)
    return offer, targeted, p_offer, y
