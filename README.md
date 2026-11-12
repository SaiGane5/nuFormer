# nuFormer-style product recommendation with Qwen3-1.7B

This project predicts which of five bank products a customer will take up (travel card, personal loan, investments, insurance, cashback plus) from their transaction history. It is a small-scale replication of Nubank's nuFormer ([arXiv:2507.23267](https://arxiv.org/abs/2507.23267)), with Qwen3-1.7B + LoRA as the sequence backbone.

## Results

The model does not beat LightGBM yet. Training on 4× more customers narrowed the gap from 3.8% to 2.5%.

| Model (test AUC) | Round 2: 13k train / 5,069 test (2×T4) | Round 3: 50k train / 74,588 test (L40S) |
| --- | --- | --- |
| LightGBM on tabular features | 0.765 | 0.789 |
| Late fusion (LLM embedding + tabular → LightGBM) | 0.757 | not run |
| Joint fusion (Qwen3 + DCNv2, end to end) | 0.736 | 0.770 |
| DCNv2 tabular only | 0.733 | 0.778 |
| Qwen3 sequence only | 0.668 | not run |
| Oracle (true propensities) | 0.867 | 0.872 |

The AUC error margin is about ±0.023 in round 2 and about ±0.006 in round 3. The two rounds use different test sets, so compare each model's gap to LightGBM within a round. The full write-up is in `report/full_report.pdf`, and the round 3 metrics and log are in `results_50k/`.

## How it works

1. `synthgen/` generates a synthetic dataset: 20k customers and ~4.25M transactions, with known true propensities. The design is in `DESIGN.md`.
2. Each transaction becomes ~12 special tokens (sign, amount bucket, month, day, weekday) plus its BPE-tokenised description. The last ~87 transactions fit in a 1,024-token context.
3. Qwen3-1.7B with LoRA (r=16 on all 7 projections, 17.4M trainable parameters) reads the sequence. A DCNv2 + PLR network reads the tabular profile, and a fusion head outputs one logit per product (BCE loss).
4. The data uses a time-disjoint split: train on months 1–8, validate on month 9, test on months 10–12.

## Run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt torch
.venv/bin/python -m synthgen.generate --n-users 20000 --out data
OUT=out_v5 DATA=data .venv/bin/python kaggle/nuformer_kaggle.py           # full run, 1 GPU
OUT=out_smoke DATA=data .venv/bin/python kaggle/nuformer_kaggle.py --smoke   # tiny CPU check
```

`kaggle/kernel-metadata.json` runs the same script as a Kaggle kernel on a T4 (about 6.4 h). An L40S takes about 1.5 h. `nuformer_kaggle_v4.py` and `results_v4/` are the earlier v4 run, kept for comparison.

## Layout

| Path | Contents |
| --- | --- |
| `synthgen/` | Synthetic data generator, labels and validation |
| `kaggle/nuformer_kaggle.py` | Tokenisation, Qwen3 + LoRA, DCNv2, fusion, training, evaluation, report |
| `kaggle/lgbm_curve.py` | LightGBM learning curve over training-set size |
| `report/` | One-pager, full report and key-concepts notes (LaTeX + PDF) |
| `results_v4/` | v4 run log, metrics and plots |
| `results_50k/` | Round 3 (50k customers) metrics, plot and log |
