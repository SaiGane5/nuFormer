"""Data-contract checks (HLD 1.4 / LLD 2.5). Raises AssertionError on violation; returns a stats dict."""
from __future__ import annotations

import numpy as np
import pandas as pd


def auc(y, s) -> float:
    y, s = np.asarray(y), np.asarray(s)
    r = pd.Series(s).rank().to_numpy()
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def validate(users: pd.DataFrame, tx: pd.DataFrame, truth: pd.DataFrame) -> dict:
    sd = users.set_index("user_id").score_date
    assert (tx.date.to_numpy() < sd.reindex(tx.user_id).to_numpy()).all(), "leakage: txn on/after score_date"
    assert tx.groupby("user_id").date.apply(lambda s: s.is_monotonic_increasing).all(), "txns not time-ordered"
    per_user = tx.groupby("user_id").size()
    assert per_user.min() >= 20, f"user with only {per_user.min()} txns"
    assert not users.drop(columns=["declared_income"]).isna().any().any(), "NaNs in users"
    assert set(users.split) == {"train", "val", "test"}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert users[users.split == a].score_date.max() < users[users.split == b].score_date.min(), f"time overlap {a}/{b}"
    rate = users.label.mean()
    assert 0.06 <= rate <= 0.25, f"label rate {rate:.3f} out of band"
    oracle = auc(users.label, users.p_true_offer)
    assert 0.70 <= oracle <= 0.95, f"oracle AUC {oracle:.3f} out of band"
    assert np.allclose(truth.groupby("user_id").size(), truth["product"].nunique())
    return {
        "n_users": int(len(users)), "n_txns": int(len(tx)),
        "txns_per_user": {"min": int(per_user.min()), "median": float(per_user.median()), "max": int(per_user.max())},
        "label_rate": round(float(rate), 4), "oracle_auc": round(oracle, 4),
        "label_rate_by_split": users.groupby("split").label.mean().round(4).to_dict(),
        "rows_by_split": users.split.value_counts().to_dict(),
        "label_rate_by_offer": users.groupby("offered_product").label.mean().round(4).to_dict(),
        "category_share": tx.category.value_counts(normalize=True).round(4).head(25).to_dict(),
    }
