# Synthetic Financial-Transaction Data for nuFormer-style Recommendation — HLD / LLD

Reference: Ramanath et al., *Your Spending Needs Attention: Modeling Financial Habits with Transformers* (Nubank, arXiv:2507.23267, 2025) — "nuFormer".

## 0. What we replicate from the paper

| Paper (nuFormer) | This project |
|---|---|
| Transaction = amount, date, description | same, plus a hidden category used only by the generator |
| Special tokens: sign (2), amount bucket (21), month (12), day (31), weekday (7), separator | identical token families; the 74 special tokens have their own trainable embedding table |
| Description tokenized with BPE | Qwen3 BPE tokenizer |
| ~14 tokens/transaction, context 512–2048 | ~13–16 tokens/txn, context 640 (T4 budget) |
| Causal GPT-style backbone pre-trained by next-token prediction | **Qwen3-1.7B** (already pre-trained) + LoRA — the user's chosen base |
| Supervised fine-tune: last non-pad token → MLP | same |
| Late fusion: fine-tuned embedding + tabular → GBT | same (LightGBM) |
| Joint fusion: transformer + DCNv2 (PLR numeric embeddings) trained end-to-end | same |
| Recommendation task: will member activate the offered product? 6-month window, disjoint time split, AUC vs LightGBM baseline | same, with a synthetic 60-day activation window and a time-disjoint split |

The real Nubank data is private, so we build a generator whose **ground truth is known**. This lets us measure two things: (a) AUC, as in the paper, and (b) ranking quality against the *true* per-product propensities, which real data can never give.

---

## 1. HLD — High-Level Design

### 1.1 Goals
1. Produce realistic per-user **transaction histories** (amount, timestamp, free-text description) together with **tabular features** and a **binary recommendation label**.
2. Plant signal so that a sequence model can, in principle, beat a tabular model. In the paper this gap exists because sequences carry recency, order and description semantics that aggregates lose.
3. Be fully reproducible (seeded), validated and scalable (vectorised; 20k users ≈ 4M transactions in about a minute on a laptop).

### 1.2 Non-goals
Matching any real bank's distributions, or privacy guarantees for real data (none is used).

### 1.3 Component view

```
              ┌──────────────────┐
  config ───► │  PersonaSampler  │  latent traits z_u (income, saver, travel, stress, family, …)
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐     ┌──────────────────┐
              │  LifeEventEngine │────►│  MerchantCatalog │ merchants × category × description templates
              └────────┬─────────┘     └────────┬─────────┘
       events in last 180 days                  │
       (raise, stress onset, trip, new car, baby)│
                       ▼                        ▼
              ┌───────────────────────────────────────┐
              │        TransactionSimulator           │ recurring (salary, rent, subs) + Poisson
              │  (per-category intensity λ_c(z,t))    │ discretionary + episodic (travel, fees)
              └────────┬──────────────────────────────┘
                       ▼
     ┌─────────────────┴──────────────────┐
     ▼                                    ▼
┌──────────────┐                 ┌──────────────────┐
│ LabelOracle  │  true p(u,k)    │ TabularFeature   │ 90-day aggregates, noisy declared income,
│ per product k│  for all 5      │ Builder          │ coarse categories (lossy by design)
└──────┬───────┘                 └────────┬─────────┘
       ▼  offer k_u ~ targeting policy    │
   label y ~ Bernoulli(p(u,k_u))          │
       └────────────────┬─────────────────┘
                        ▼
              ┌──────────────────┐     ┌────────────┐
              │ Splitter (time)  │────►│ Validator  │ schema, ranges, base rates, leakage checks
              └────────┬─────────┘     └────────────┘
                       ▼
              Parquet: users.parquet, transactions.parquet, product_truth.parquet, meta.json
```

### 1.4 Data flow and contracts
* **users.parquet**: one row per (user, score_date, offered_product). It holds the tabular features, `label`, `split` ∈ {train, val, test} and `score_date`.
* **transactions.parquet**: `user_id, ts, amount (signed), description`. Only rows strictly **before** `score_date` (no leakage).
* **product_truth.parquet**: `user_id, product, p_true`, the oracle propensity for *every* product. Used only for evaluation.
* **meta.json**: generator version, seed, config hash, row counts, base rates.

### 1.5 Signal design (why sequences should help)
The label logit for product *k* combines:
1. **Static traits** that the tabular features capture partly and noisily (income, age, tenure).
2. **Behavioural traits** visible as *merchant/description semantics* (airline and hotel names → travel; brokerage transfers → saver). Tabular features only hold 8 coarse category sums.
3. **Recency effects**: a life event in the last 30 days (stress onset, salary raise, new car, newborn) shifts propensity strongly. 90-day aggregates dilute these; the sequence shows them in order at the end of the context.
4. Irreducible noise, so the task is not trivially separable (target oracle AUC ≈ 0.80–0.85).

### 1.6 Split strategy (paper: disjoint train/test periods)
`score_date` is spread over 12 months. Months 1–8 are train, month 9 is val, months 10–12 are test. Users are also disjoint across splits, so there is no history overlap.

### 1.7 Scale and budget (Kaggle T4, 16 GB)
20k scoring rows (train ~13.3k, val ~1.7k, test ~5k). Context 640 tokens ≈ the 40–45 most recent transactions.

---

## 2. LLD — Low-Level Design

### 2.1 Modules (`synthgen/`)
| Module | Key types / functions | Notes |
|---|---|---|
| `config.py` | `@dataclass(frozen=True) GenConfig` | every knob in one place; `config_hash()` for lineage |
| `catalog.py` | `Category` enum, `Merchant(name, category, amount_mu, amount_sigma, templates)`, `CATALOG` | ~120 merchants, 16 categories; templates like `"UBER *TRIP {city}"`, `"PIX TRANSF {name}"` |
| `personas.py` | `sample_personas(n, rng) -> pd.DataFrame` | latent traits, correlated via a Gaussian copula (income ↔ saver, stress ↔ −income) |
| `events.py` | `sample_life_events(personas, rng)` | ≤1 event/user, with type and day offset; P(event in last 30d) ≈ 0.35 |
| `simulator.py` | `simulate(personas, events, cfg, rng) -> transactions df` | vectorised per category: recurring schedules, Poisson counts, log-normal amounts, event-driven bursts |
| `labels.py` | `PRODUCTS`, `true_logits(personas, events, txn_stats) -> (n,5)`, `assign_offers`, `draw_labels` | product logits are linear in latents plus interaction terms; calibrated intercepts give a base rate of ~12% |
| `features.py` | `build_tabular(personas, txns, score_date)` | 28 features; deliberately lossy (coarse categories, noisy income, 90-day window) |
| `validate.py` | `validate(users, txns, truth)` | asserts: no txn ≥ score_date, label rate in [8%, 20%], every user has ≥ 20 txns, monotone ts, no NaN, oracle AUC within band |
| `generate.py` | CLI `python -m synthgen.generate --n-users 20000 --seed 7 --out data/` | writes parquet + meta.json |

### 2.2 Key algorithms
**Recurring flows.** Salary is paid on day `d_s ∈ {1,5,15,30}` (or biweekly). Its amount is `income_monthly × N(1, 0.02)`, and it rises by 15–40% after a `RAISE` event. Rent and utilities are paid monthly on a fixed day. Each user has 0–4 subscriptions, chosen with probability by trait (for example, streaming is more likely when `digital` is high).

**Discretionary flows.** For each category *c* and day *t*: `N_c(t) ~ Poisson(λ_c · g_c(z_u) · h_c(event, t))`. Here `g_c` is a trait multiplier and `h_c` is an event bump (for example, a `NEW_CAR` event adds auto-parts and fuel purchases after it happens). Amounts follow `LogNormal(μ_m + β·log income, σ_m)` for each merchant.

**Episodic flows.** Travel trips cluster within a 3–10 day window: airline, then hotel, then FX and restaurants abroad. `STRESS` episodes produce late fees, overdraft charges, payday-lender inflows and collection agencies, with frequency growing toward the score date.

**Descriptions.** A template is filled with a city, a store number (`#0423`) or a counterpart name. Uppercase banking style is applied, with 5% random truncation to mimic real statement noise.

**Label oracle.** Each product *k* has a logit over z-scored latents:
```
TRAVEL_CARD    = a0 + 1.6·travel + 0.6·income − 0.5·stress
PERSONAL_LOAN  = a1 + 1.4·stress + 1.2·[STRESS event ≤30d] − 0.6·saver
INVESTMENTS    = a2 + 1.5·saver + 0.7·income + 0.9·[RAISE ≤30d] − 0.8·stress
INSURANCE      = a3 + 1.0·family + 1.3·[NEW_CAR or BABY ≤45d] + 0.3·age
CASHBACK_PLUS  = a4 + 1.2·discretionary_share + 0.5·digital
```
plus N(0, 0.5) noise. The offer policy is 70% random and 30% "targeted" (the product with the largest noisy logit), mimicking the bias of a real campaign. Labels are drawn as `y ~ Bernoulli(sigmoid(logit[k_offer]))`.

**Amount bucketing (consumer side, same as the paper).** 21 buckets come from quantiles of `log1p(|amount|)` fitted **on train only**; the sign is a separate token.

### 2.3 Training-side tokenization (`nuformer_kaggle.py`)
```
<TX> [SIGN±] [AMT_b] [MON_m] [DAY_d] [WD_w] <bpe(description)> ... <TX> ...
```
* The 74 special tokens are mapped to an **auxiliary trainable embedding** (74 × d_model). It is added in place of the base embedding at special positions via `inputs_embeds`. This avoids resizing Qwen's 151k vocab, which would require training a 311M-parameter matrix.
* Histories are left-truncated to the most recent transactions that fit in the 640-token context, and left-padded.
* The embedding is the hidden state of the final token (left padding means the last position is always real).

### 2.4 Models trained (mirrors paper Table 4)
1. LightGBM on tabular only (the baseline).
2. DCNv2 + PLR on tabular only.
3. Qwen3-1.7B **frozen** embedding + tabular → LightGBM (no fine-tuning).
4. Sequence-only LoRA fine-tune (Qwen + MLP head).
5. **Late fusion**: model 4's embedding + tabular → LightGBM.
6. **nuFormer joint fusion**: LoRA-Qwen embedding ⊕ DCNv2(PLR) → MLP, trained end-to-end with BCE.

Metrics: test AUC, relative AUC lift vs. LightGBM (the paper's headline metric), PR-AUC, and ranking against the oracle (NDCG@1/@3 and Hit@1 of the true-best product, scoring all 5 products per test user).

### 2.5 Failure modes and guards
| Risk | Guard |
|---|---|
| Leakage via future txns | validator asserts `ts < score_date` |
| Leakage via label in tabular features | features are computed from txns + personas only; label is drawn afterwards |
| Too-easy task | oracle AUC band check; noise term |
| Amount-bucket drift | bucket edges fit on train, saved to meta |
| T4 OOM | grad checkpointing, fp16 autocast, batch 4 × accumulation 8, context 640 |
