"""LightGBM learning curve on nested training subsets (CPU only, minutes).

DATA=data_300k N_VAL=5000 python kaggle/lgbm_curve.py 13265 25000 50000 100000 0
Subsets are drawn exactly like nuformer_kaggle.py (seed 42, one permutation per split, first N), so the test
set and the training customers match the GPU runs for the same N_TRAIN.  0 = all training customers.
"""
import json, os, sys, time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

SEED = 42
PRODUCTS = ["TRAVEL_CARD", "PERSONAL_LOAN", "INVESTMENTS", "INSURANCE", "CASHBACK_PLUS"]
CAT_COLS = ["region", "channel", "offered_product"]
DROP = {"user_id", "score_date", "score_month", "split", "label", "p_true_offer", "targeted_offer"}

DATA = os.environ.get("DATA", "data_300k")
n_val, n_test = int(os.environ.get("N_VAL", "0")), int(os.environ.get("N_TEST", "0"))
sizes = [int(s) for s in sys.argv[1:]] or [0]

users = pd.read_parquet(f"{DATA}/users.parquet")
rs = np.random.default_rng(SEED)
perm = {s: rs.permutation(users.user_id[users.split == s].to_numpy()) for s in ("train", "val", "test")}
va_ids = perm["val"][:n_val] if n_val else perm["val"]
te_ids = perm["test"][:n_test] if n_test else perm["test"]
U = users.set_index("user_id")
feat = [c for c in users.columns if c not in DROP]
X = U[feat].copy()
for c in CAT_COLS:
    X[c] = pd.Categorical(X[c], categories=PRODUCTS if c == "offered_product" else sorted(users[c].unique()))
y = U.label

Xva, yva, Xte, yte = X.loc[np.sort(va_ids)], y.loc[np.sort(va_ids)], X.loc[np.sort(te_ids)], y.loc[np.sort(te_ids)]
out = {"data": DATA, "n_val": len(va_ids), "n_test": len(te_ids), "test_pos_rate": round(float(yte.mean()), 4),
       "oracle_test_auc": round(float(roc_auc_score(yte, U.p_true_offer.loc[yte.index])), 4), "curve": []}
print(json.dumps({k: v for k, v in out.items() if k != "curve"}), flush=True)
for n in sizes:
    tr_ids = np.sort(perm["train"][:n] if n else perm["train"])
    t = time.time()
    m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=-1)
    m.fit(X.loc[tr_ids], y.loc[tr_ids], eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(150, verbose=False)])
    p = m.predict_proba(Xte)[:, 1]
    r = {"n_train": len(tr_ids), "test_auc": round(float(roc_auc_score(yte, p)), 4),
         "test_pr_auc": round(float(average_precision_score(yte, p)), 4), "trees": int(m.best_iteration_ or 0), "sec": round(time.time() - t, 1)}
    out["curve"].append(r); print(json.dumps(r), flush=True)
json.dump(out, open(f"lgbm_curve_{os.path.basename(DATA)}.json", "w"), indent=2)
