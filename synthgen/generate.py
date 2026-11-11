"""CLI: python -m synthgen.generate --n-users 20000 --seed 7 --out data/"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from .config import GenConfig
from .features import build_tabular
from .labels import assign_offers_and_labels, true_logits
from .personas import sample_life_events, sample_personas
from .simulator import simulate
from .validate import validate

__version__ = "1.0.0"


def generate(cfg: GenConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    p = sample_personas(cfg, rng)
    ev = sample_life_events(p, cfg, rng)
    month = rng.integers(0, cfg.n_months, cfg.n_users)
    score_dates = (np.datetime64(cfg.base_date) + (month * 30 + rng.integers(0, 30, cfg.n_users)).astype("timedelta64[D]"))
    tx = simulate(p, ev, score_dates, cfg, rng)
    L = true_logits(p, ev, tx, cfg, rng)
    offer, targeted, p_offer, y = assign_offers_and_labels(L, cfg, rng)
    feats = build_tabular(p, tx, score_dates, cfg, rng)

    split = np.select([np.isin(month, cfg.train_months), np.isin(month, cfg.val_months)], ["train", "val"], "test")
    users = feats.assign(score_date=score_dates, score_month=month, split=split,
                         offered_product=np.array(cfg.products)[offer], targeted_offer=targeted.astype(int),
                         label=y, p_true_offer=p_offer.round(5))
    # keep generator-only latents in a separate file for analysis (never used as model inputs)
    latents = p[["user_id"] + [c for c in p.columns if c.startswith("z_")] + ["income_monthly", "has_kids", "has_car"]].merge(ev, on="user_id")
    truth = pd.DataFrame({"user_id": np.repeat(p.user_id.to_numpy(), len(cfg.products)),
                          "product": np.tile(cfg.products, cfg.n_users),
                          "p_true": (1 / (1 + np.exp(-L))).ravel().round(5)})
    return {"users": users, "transactions": tx, "truth": truth, "latents": latents}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-users", type=int, default=GenConfig.n_users)
    ap.add_argument("--seed", type=int, default=GenConfig.seed)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    cfg = replace(GenConfig(), n_users=a.n_users, seed=a.seed)
    t0 = time.time()
    d = generate(cfg)
    stats = validate(d["users"], d["transactions"], d["truth"])
    os.makedirs(a.out, exist_ok=True)
    d["users"].to_parquet(f"{a.out}/users.parquet", index=False)
    d["transactions"].to_parquet(f"{a.out}/transactions.parquet", index=False)
    d["truth"].to_parquet(f"{a.out}/product_truth.parquet", index=False)
    d["latents"].to_parquet(f"{a.out}/latents.parquet", index=False)
    meta = {"generator_version": __version__, "config": asdict(cfg), "config_hash": cfg.config_hash(),
            "seconds": round(time.time() - t0, 1), **stats}
    json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=2, default=str)
    print(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()
