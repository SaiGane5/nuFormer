"""All generator knobs in one frozen dataclass (LLD 2.1)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

PRODUCTS = ("TRAVEL_CARD", "PERSONAL_LOAN", "INVESTMENTS", "INSURANCE", "CASHBACK_PLUS")


@dataclass(frozen=True)
class GenConfig:
    n_users: int = 20_000
    seed: int = 7
    history_days: int = 150          # transactions in [score_date - history_days, score_date)
    base_date: str = "2025-01-01"    # month 0 of the scoring calendar
    n_months: int = 12               # score dates spread over 12 months
    train_months: tuple = (0, 1, 2, 3, 4, 5, 6, 7)
    val_months: tuple = (8,)
    test_months: tuple = (9, 10, 11)
    target_base_rate: float = 0.08   # per-product mean activation propensity
    label_noise_sd: float = 0.5      # irreducible noise on the logit
    targeted_offer_share: float = 0.30
    p_life_event: float = 0.60
    median_monthly_income: float = 4_000.0
    desc_truncate_p: float = 0.05
    regions: tuple = ("SE", "S", "NE", "CO", "N")
    channels: tuple = ("PUSH", "EMAIL", "IN_APP")
    products: tuple = field(default=PRODUCTS)

    def config_hash(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True, default=str).encode()).hexdigest()[:12]
