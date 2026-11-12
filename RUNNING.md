# How to run each module

All commands run from the repository root (the folder that holds `synthgen/` and `kaggle/`).

## 0. Setup (once)

Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt torch
```

The first model run downloads the Qwen3-1.7B tokenizer (and, on a GPU, the ~3.4 GB weights) from Hugging Face. On Kaggle the script uses the attached model instead, with no internet.

## 1. Synthetic data generator — `synthgen/`

Builds customers, transactions, true propensities and the tabular profile, then runs the validator. The validator stops with an `AssertionError` if any data check fails.

```bash
# quick look: 2,000 customers, a few seconds
python -m synthgen.generate --n-users 2000 --seed 7 --out data_2k

# the dataset used for rounds 1-2: 20,000 customers, ~4.25M transactions
python -m synthgen.generate --n-users 20000 --seed 7 --out data

# large datasets, generated in chunks to keep memory bounded (round 3 used 300k)
python -m synthgen.generate_big --n-users 300000 --chunk 50000 --seed 7 --out data_300k
```

Each output folder holds:

| File | Contents |
| --- | --- |
| `users.parquet` | One row per customer: tabular features, offered product, label, split (train / val / test) |
| `transactions.parquet` | Every transaction: date, amount, description, merchant, category |
| `product_truth.parquet` | True take-up probability for each customer × product |
| `latents.parquet` | Hidden traits and life events (for analysis only, never a model input) |
| `meta.json` | Config, config hash, run time and validator statistics (label rate, oracle AUC, ...) |

The modules inside `synthgen/` are not run on their own; `generate.py` calls them in this order:

| Module | Job |
| --- | --- |
| `config.py` | Every generator setting in one place (`GenConfig`) |
| `personas.py` | Correlated hidden traits and one life event per customer |
| `catalog.py` | 61 merchants, amounts, price tiers, statement description templates |
| `simulator.py` | Recurring, daily Poisson and trip transactions |
| `labels.py` | True per-product propensities, offers and accept / decline labels |
| `features.py` | The deliberately lossy tabular profile |
| `validate.py` | Data checks (no leakage, label rate, oracle AUC range) |

## 2. Main model — `kaggle/nuformer_kaggle.py`

One script does everything: tokenise histories, fine-tune Qwen3-1.7B with LoRA, train DCNv2, fit LightGBM and the fusion models, then write the report.

```bash
# CPU smoke test (under a minute): tiny random model, 120 customers per split
OUT=out_smoke DATA=data_2k LGBM_JOBS=1 python kaggle/nuformer_kaggle.py --smoke

# full run on one GPU (L40S ~1.5 h, T4 ~6.4 h)
OUT=out_v5 DATA=data python kaggle/nuformer_kaggle.py

# round 3: 50k training customers, joint model only
OUT=out_50k DATA=data_300k N_TRAIN=50000 N_VAL=5000 MODES=joint python kaggle/nuformer_kaggle.py
```

The smoke test only checks that the pipeline runs end to end; its AUCs mean nothing.

On macOS keep `LGBM_JOBS=1`. Without it the report step crashes with a segmentation fault (exit code 139) after both workers finish, because LightGBM's and PyTorch's OpenMP runtimes clash. Linux GPU machines don't need it.

Settings you can change through environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DATA` | `data` | Folder made by the generator |
| `OUT` | `/kaggle/working` | Where results go — always set it when running locally |
| `MODES` | `seq,joint` | Which fine-tunes to run (`seq` = Qwen3 only, `joint` = Qwen3 + DCNv2) |
| `N_TRAIN`, `N_VAL`, `N_TEST` | `0` (all) | Number of customers to use from each split |
| `CTX` | `1024` | Context length in tokens |
| `BATCH`, `ACCUM` | `4`, `8` | Batch size and gradient accumulation steps |
| `EPOCHS`, `MAX_TRAIN_HOURS` | `1`, `5.5` | Training length and a hard time limit |
| `GRAD_CKPT` | `1` | Gradient checkpointing (set `0` on large GPUs for speed) |
| `FP16` | unset | Set `1` to force fp16 instead of bf16 |
| `LGBM_JOBS` | `-1` (all cores) | LightGBM threads in the report step (use `1` on macOS) |

With two GPUs and two modes, the two fine-tunes run in parallel, one per GPU. Outputs in `OUT`:

| File | Contents |
| --- | --- |
| `results.json` | Test AUC, PR-AUC and lift over LightGBM for every model, plus the oracle |
| `results.png` | AUC bar chart and training-loss curves |
| `worker_<mode>.json` | Per-fine-tune metrics and validation curve |

To run it as a Kaggle kernel instead (T4, needs the Kaggle CLI and the dataset uploaded to your account):

```bash
kaggle kernels push -p kaggle
kaggle kernels status hellofriend375/nuformer-qwen3-reco
kaggle kernels output hellofriend375/nuformer-qwen3-reco -p out_kaggle
```

Edit the `id` and `dataset_sources` in `kaggle/kernel-metadata.json` to match your own Kaggle account first.

## 3. LightGBM learning curve — `kaggle/lgbm_curve.py`

CPU only, a few minutes. Trains LightGBM on growing training subsets, using the same customer sampling as the GPU runs. `0` means all training customers.

```bash
DATA=data_300k N_VAL=5000 python kaggle/lgbm_curve.py 13265 25000 50000 100000 0
```

It prints one JSON line per size and writes `lgbm_curve_<data folder>.json` to the current folder.

## 4. Earlier v4 script — `nuformer_kaggle_v4.py`

Kept to reproduce `results_v4/` (512-token context, older output head). It is run the same way as the main script:

```bash
OUT=out_v4 DATA=data python nuformer_kaggle_v4.py
```

## 5. Reports — `report/`

Built with [tectonic](https://tectonic-typesetting.github.io) (`brew install tectonic`), which fetches LaTeX packages on first use.

```bash
cd report
tectonic full_report.tex        # 9-page write-up -> full_report.pdf
tectonic nuformer_onepager.tex  # one-page summary -> nuformer_onepager.pdf
tectonic key_concepts.tex       # concept notes -> key_concepts.pdf
```

`report/data-generator.html` is a self-contained page explaining the generator; open it in any browser.

## 6. PaySim fraud fine-tune — `../kaggle_qwen_finetune/` (separate experiment)

Fine-tunes Qwen3-1.7B with LoRA to label PaySim transactions as fraud or legit. It reads its inputs from `/kaggle/input`, so it only runs as a Kaggle kernel:

```bash
kaggle kernels push -p kaggle_qwen_finetune
kaggle kernels status saigane/qwen3-1-7b-paysim-fraud-lora
kaggle kernels output saigane/qwen3-1-7b-paysim-fraud-lora -p out_paysim
```

Run these from the folder above this repository, the one that holds `kaggle_qwen_finetune/`.
