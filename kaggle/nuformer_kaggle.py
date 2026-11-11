# nuFormer-style recommendation fine-tuning of Qwen3-1.7B on synthetic transaction data.
# Paper: Ramanath et al., "Your Spending Needs Attention" (Nubank, arXiv:2507.23267).
#
# Phases (single Kaggle script, T4 x2):
#   prep   : tokenize histories (sign/amount/date special tokens + BPE description), fit amount buckets on train
#   workers: [GPU0] sequence(+offer) LoRA classifier     [GPU1] nuFormer joint fusion (LoRA + DCNv2/PLR)
#            each also dumps embeddings; GPU0 first dumps frozen (no-finetune) embeddings
#   report : LightGBM tabular, DCNv2 tabular, frozen-emb fusion, late fusion, joint fusion; AUC + oracle ranking
import os, sys, glob, json, time, math, argparse, subprocess, random

ap = argparse.ArgumentParser()
ap.add_argument("--worker", default=None)        # None | seq | joint
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--smoke", action="store_true")  # tiny CPU run for local testing
ARGS, _ = ap.parse_known_args()
if ARGS.worker:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.gpu)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

SMOKE = ARGS.smoke or os.environ.get("SMOKE") == "1"
OUT = os.environ.get("OUT", "/kaggle/working")
DATA = os.environ.get("DATA") or (os.path.dirname(glob.glob("/kaggle/input/**/users.parquet", recursive=True)[0])
                                  if glob.glob("/kaggle/input/**/users.parquet", recursive=True) else "data")
CFG = dict(ctx=1024, desc_max_tokens=6, batch=4, accum=8, epochs=1, lr_lora=2e-4, lr_head=1e-3, wd_dcn=1e-4, lora_r=16, lora_alpha=32,
           evals_per_epoch=4, max_train_hours=5.5, n_amount_buckets=21, seed=42, frozen_baseline=False)
# v5: ctx 512->1024 and descriptions capped at 6 BPE tokens (~2.4x more history); per-product output head
# scale runs: every knob below can be overridden from the environment (defaults reproduce v5 exactly)
for _k, _cast in (("ctx", int), ("batch", int), ("accum", int), ("epochs", int), ("max_train_hours", float), ("evals_per_epoch", int)):
    if os.environ.get(_k.upper()): CFG[_k] = _cast(os.environ[_k.upper()])
CFG.update(grad_ckpt=os.environ.get("GRAD_CKPT", "1") == "1", eval_bs=int(os.environ.get("EVAL_BS", "16")),
           n_train=int(os.environ.get("N_TRAIN", "0")), n_val=int(os.environ.get("N_VAL", "0")), n_test=int(os.environ.get("N_TEST", "0")),
           modes=os.environ.get("MODES", "seq,joint"))
# bf16 where the GPU supports it (Ampere+: no loss scaling needed); fp16 + GradScaler on T4
AMP = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported() and os.environ.get("FP16") != "1") else torch.float16
if SMOKE:
    CFG.update(ctx=128, batch=4, accum=1, evals_per_epoch=1, max_train_hours=0.2)
SEED = CFG["seed"]
PRODUCTS = ["TRAVEL_CARD", "PERSONAL_LOAN", "INVESTMENTS", "INSURANCE", "CASHBACK_PLUS"]
CAT_COLS = ["region", "channel", "offered_product"]
DROP = {"user_id", "score_date", "score_month", "split", "label", "p_true_offer", "targeted_offer"}
MONEY = ["declared_income", "credit_limit", "inflow_90d", "outflow_90d"]  # + spend_*_90d


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def find_model_dir():
    if SMOKE:
        return None
    c = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/config.json", recursive=True)
         if "1.7b" in p.lower() and "base" not in p.lower()]
    return c[0] if c else "Qwen/Qwen3-1.7B"


# ============================================================================ special-token vocabulary (paper §3)
def special_vocab():
    names = ["<TX>"] + ["<SIGN_IN>", "<SIGN_OUT>"] + [f"<AMT_{i}>" for i in range(CFG["n_amount_buckets"])]
    names += [f"<MON_{i}>" for i in range(1, 13)] + [f"<DAY_{i}>" for i in range(1, 32)] + [f"<WD_{i}>" for i in range(7)]
    return names


WD = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MON = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]


def special_init_text(name, edges):
    """Text rendering used to initialise each special-token embedding from the base vocab (mean of its BPE pieces)."""
    if name == "<TX>": return "\n transaction:"
    if name == "<SIGN_IN>": return " credit received"
    if name == "<SIGN_OUT>": return " debit paid"
    k, v = name[1:-1].split("_")
    v = int(v)
    if k == "AMT":
        lo = 0 if v == 0 else math.expm1(edges[v - 1]); return f" amount about {int(lo)} reais"
    if k == "MON": return f" {MON[v - 1]}"
    if k == "DAY": return f" day {v}"
    if k == "WD": return f" {WD[v]}"


# ============================================================================ PREP
def prep():
    from transformers import AutoTokenizer
    t0 = time.time()
    users = pd.read_parquet(f"{DATA}/users.parquet")
    caps = {"train": CFG["n_train"], "val": CFG["n_val"], "test": CFG["n_test"]}
    if any(caps.values()):
        # nested subsets: a fixed permutation per split, take the first `cap` (so 50k train is inside 100k, test is identical)
        rs, keep = np.random.default_rng(SEED), []
        for s, cap in caps.items():
            ids = rs.permutation(users.user_id[users.split == s].to_numpy())
            keep.append(ids[:cap] if cap else ids)
        keep = np.sort(np.concatenate(keep)); users = users[users.user_id.isin(keep)]
        flt = [("user_id", "in", keep.tolist())]
        tx = pd.read_parquet(f"{DATA}/transactions.parquet", filters=flt)
        truth = pd.read_parquet(f"{DATA}/product_truth.parquet", filters=flt)
    else:
        tx = pd.read_parquet(f"{DATA}/transactions.parquet")
        truth = pd.read_parquet(f"{DATA}/product_truth.parquet")
    if SMOKE:
        keep = users.groupby("split").head(120).user_id
        users, tx, truth = users[users.user_id.isin(keep)], tx[tx.user_id.isin(keep)], truth[truth.user_id.isin(keep)]
    users = users.sort_values("user_id").reset_index(drop=True)
    log("users", users.shape, "tx", tx.shape)

    tok = AutoTokenizer.from_pretrained(find_model_dir() or "Qwen/Qwen3-1.7B")
    specials = special_vocab()
    S = {n: i for i, n in enumerate(specials)}
    V0 = 1 << 20  # special ids live at V0+k (never collide with base vocab; mapped to aux embedding in the model)

    # amount buckets: quantiles of log1p|amount| on TRAIN users only (no leakage)
    la = np.log1p(tx.amount.abs().to_numpy())
    train_ids = set(users.user_id[users.split == "train"])
    tr_mask = tx.user_id.isin(train_ids).to_numpy()
    edges = np.quantile(la[tr_mask], np.linspace(0, 1, CFG["n_amount_buckets"] + 1)[1:-1])
    bucket = np.searchsorted(edges, la, side="right")

    # BPE for descriptions (cache unique strings)
    uniq = pd.unique(tx.description)
    enc = tok(list(uniq), add_special_tokens=False).input_ids
    desc_ids = dict(zip(uniq, enc))
    d = pd.to_datetime(tx.date)
    sign = np.where(tx.amount.to_numpy() > 0, S["<SIGN_IN>"], S["<SIGN_OUT>"])
    head = np.stack([np.full(len(tx), S["<TX>"]), sign, S["<AMT_0>"] + bucket, S["<MON_1>"] + d.dt.month.to_numpy() - 1,
                     S["<DAY_1>"] + d.dt.day.to_numpy() - 1, S["<WD_0>"] + d.dt.weekday.to_numpy()], 1) + V0
    descs = tx.description.to_numpy()
    uids = tx.user_id.to_numpy()

    # per-user sequences: most recent transactions that fit in ctx (paper: left-truncate history)
    ctx = CFG["ctx"]
    starts = np.r_[0, np.nonzero(np.diff(uids))[0] + 1]; ends = np.r_[starts[1:], len(uids)]
    seqs, n_tx_in_ctx, tok_per_tx = {}, [], []
    for s, e in zip(starts, ends):
        pieces, total = [], 0
        for i in range(e - 1, s - 1, -1):
            p = list(head[i]) + desc_ids[descs[i]][:CFG["desc_max_tokens"]]
            if total + len(p) > ctx: break
            pieces.append(p); total += len(p); tok_per_tx.append(len(p))
        n_tx_in_ctx.append(len(pieces))
        seqs[int(uids[s])] = np.array([t for p in reversed(pieces) for t in p], dtype=np.int64)
    seq_list = [seqs[int(u)] for u in users.user_id]

    # tabular matrix (numeric standardised on train; money log-transformed) + categorical codes
    num_cols = [c for c in users.columns if c not in DROP and c not in CAT_COLS]
    Xn = users[num_cols].astype(float).copy()
    for c in num_cols:
        if c in MONEY or c.startswith("spend_"):
            Xn[c] = np.sign(Xn[c]) * np.log1p(Xn[c].abs())
    tr = (users.split == "train").to_numpy()
    mu, sd = Xn[tr].mean(), Xn[tr].std().replace(0, 1)
    Xn = ((Xn - mu) / sd).to_numpy(np.float32)
    cats = {c: sorted(users[c].unique()) if c != "offered_product" else PRODUCTS for c in CAT_COLS}
    Xc = np.stack([users[c].map({v: i for i, v in enumerate(cats[c])}).to_numpy() for c in CAT_COLS], 1).astype(np.int64)

    init_text = [special_init_text(n, edges) for n in specials]
    init_ids = [tok(t, add_special_tokens=False).input_ids for t in init_text]
    stats = {"prep_sec": round(time.time() - t0, 1), "n_users": len(users), "n_tx": int(len(tx)),
             "unique_descriptions": int(len(uniq)), "tokens_per_tx_mean": round(float(np.mean(tok_per_tx)), 2),
             "tx_in_context_mean": round(float(np.mean(n_tx_in_ctx)), 1), "tx_in_context_median": float(np.median(n_tx_in_ctx)),
             "seq_len_mean": round(float(np.mean([len(s) for s in seq_list])), 1), "amount_edges_log1p": edges.round(3).tolist(),
             "example_tokens": [specials[t - V0] if t >= V0 else tok.decode([t]) for t in seq_list[0][-40:]]}
    torch.save({"seqs": seq_list, "Xn": Xn, "Xc": Xc, "num_cols": num_cols, "cats": cats,
                "y": users.label.to_numpy(np.float32), "split": users.split.to_numpy(), "user_id": users.user_id.to_numpy(),
                "p_true_offer": users.p_true_offer.to_numpy(), "specials": specials, "init_ids": init_ids, "V0": V0,
                "pad_id": tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
                "users": users, "truth": truth}, f"{OUT}/prep.pt")
    log("prep done", stats)
    return stats


# ============================================================================ MODEL PARTS
class PLR(nn.Module):
    """Periodic-Linear-ReLU numeric embeddings (Gorishniy et al. 2022), as used by nuFormer's tabular branch."""
    def __init__(self, n, k=16, d=16, sigma=1.0):
        super().__init__()
        self.c = nn.Parameter(torch.randn(n, k) * sigma)
        self.lin = nn.Parameter(torch.randn(n, 2 * k, d) / math.sqrt(2 * k)); self.b = nn.Parameter(torch.zeros(n, d))
    def forward(self, x):                                 # (B,n) -> (B,n*d)
        v = 2 * math.pi * self.c[None] * x[..., None]
        v = torch.cat([torch.sin(v), torch.cos(v)], -1)   # (B,n,2k)
        return F.relu(torch.einsum("bnk,nkd->bnd", v, self.lin) + self.b).flatten(1)


class DCNv2(nn.Module):
    def __init__(self, n_num, cat_sizes, d=16, n_cross=3, hidden=256, out=64, drop=0.1):
        super().__init__()
        self.plr = PLR(n_num, d=d)
        self.embs = nn.ModuleList([nn.Embedding(s, d) for s in cat_sizes])
        D = n_num * d + len(cat_sizes) * d
        self.cross = nn.ModuleList([nn.Linear(D, D) for _ in range(n_cross)])
        self.deep = nn.Sequential(nn.Linear(D, hidden), nn.ReLU(), nn.Dropout(drop), nn.Linear(hidden, out), nn.ReLU())
    def forward(self, xn, xc):
        x0 = torch.cat([self.plr(xn)] + [e(xc[:, i]) for i, e in enumerate(self.embs)], 1)
        x = x0
        for lin in self.cross:
            x = x0 * lin(x) + x                            # DCN-v2 full-rank cross layer
        return self.deep(x)


class SeqEncoder(nn.Module):
    """Qwen3 backbone (+LoRA) with an auxiliary trainable embedding table for the 74 transaction special tokens."""
    def __init__(self, prep_obj, lora=True, device="cuda"):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoConfig
        dtype = AMP if device == "cuda" else torch.float32
        md = find_model_dir()
        if md is None:  # smoke: tiny random Qwen3 with the real tokenizer vocab
            cfg = AutoConfig.from_pretrained("Qwen/Qwen3-1.7B"); cfg.update(dict(hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16))
            lm = AutoModelForCausalLM.from_config(cfg).to(dtype)
        else:
            lm = AutoModelForCausalLM.from_pretrained(md, torch_dtype=dtype)
        self.backbone = lm.model.to(device)                # drop the 151k-vocab LM head: we only need hidden states
        del lm
        for p in self.backbone.parameters(): p.requires_grad_(False)
        if lora:
            from peft import LoraConfig, get_peft_model
            # Kaggle image ships torchao 0.10 and peft raises on it; we never use torchao, so report it as absent.
            import importlib
            for _m in ("peft.import_utils", "peft.utils.quantization_utils", "peft.tuners.lora.torchao"):
                try:
                    setattr(importlib.import_module(_m), "is_torchao_available", lambda: False)
                except Exception:
                    pass
            self.backbone = get_peft_model(self.backbone, LoraConfig(r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"], lora_dropout=0.05,
                                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
            for p in self.backbone.parameters():
                if p.requires_grad: p.data = p.data.float()
            if CFG["grad_ckpt"]: self._inner().gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.V0, self.pad_id = prep_obj["V0"], prep_obj["pad_id"]
        emb = self._inner().get_input_embeddings()
        init = torch.stack([emb.weight[torch.tensor(ids, device=emb.weight.device)].float().mean(0) for ids in prep_obj["init_ids"]])
        self.special = nn.Embedding(len(prep_obj["specials"]), init.shape[1]).to(device)
        self.special.weight.data.copy_(init)
        self.hidden = init.shape[1]

    def _inner(self):
        b = self.backbone
        return b.base_model.model if hasattr(b, "base_model") and hasattr(b.base_model, "model") else b

    def forward(self, ids, att):
        sp = ids >= self.V0
        base = torch.where(sp, torch.full_like(ids, self.pad_id), ids)
        emb = self._inner().get_input_embeddings()(base)
        spe = self.special((ids - self.V0).clamp(min=0)).to(emb.dtype)
        x = torch.where(sp[..., None], spe, emb)
        h = self.backbone(inputs_embeds=x, attention_mask=att).last_hidden_state
        return h[:, -1].float()                            # left padding => last position is the final real token


class Head(nn.Module):
    """mode=seq: [LN(h) , onehot(offer)] -> MLP.   mode=joint: [LN(h)->128 , DCNv2(tab)] -> MLP (nuFormer joint fusion)."""
    def __init__(self, hidden, mode, n_num=0, cat_sizes=()):
        super().__init__()
        self.mode, self.ln = mode, nn.LayerNorm(hidden)
        self.proj = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU())
        # v5: the offer is no longer an input; the last layer has one logit per product (loss on the offered one)
        side = 0
        if mode == "joint":
            self.dcn = DCNv2(n_num, cat_sizes[:2]); side = 64
        self.mlp = nn.Sequential(nn.Linear(128 + side, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, len(PRODUCTS)))
    def all_logits(self, h, xn, xc):                       # (B, n_products)
        z = self.proj(self.ln(h))
        if self.mode == "joint": z = torch.cat([z, self.dcn(xn, xc[:, :2])], 1)
        return self.mlp(z)
    def forward(self, h, xn, xc):
        return self.all_logits(h, xn, xc).gather(1, xc[:, 2:3]).squeeze(1)


def batches(P, idx, bs, shuffle, rng=None):
    idx = np.array(idx)
    if shuffle: rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        L = max(len(P["seqs"][j]) for j in b)
        ids = torch.full((len(b), L), P["pad_id"], dtype=torch.long); att = torch.zeros((len(b), L), dtype=torch.long)
        for r, j in enumerate(b):
            s = torch.from_numpy(P["seqs"][j]); ids[r, L - len(s):] = s; att[r, L - len(s):] = 1
        yield b, ids, att, torch.from_numpy(P["Xn"][b]), torch.from_numpy(P["Xc"][b]), torch.from_numpy(P["y"][b])


@torch.no_grad()
def embed_all(enc, P, dev, bs=None):
    bs = bs or CFG["eval_bs"]
    enc.eval(); out = np.zeros((len(P["seqs"]), enc.hidden), np.float32)
    order = np.argsort([len(s) for s in P["seqs"]])            # length-sorted for speed
    for b, ids, att, *_ in batches(P, order, bs, False):
        with torch.autocast("cuda", dtype=AMP, enabled=dev == "cuda"):
            out[b] = enc(ids.to(dev), att.to(dev)).cpu().numpy()
    return out


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


@torch.no_grad()
def score(enc, head, P, idx, dev, bs=None, all_products=False):
    bs = bs or CFG["eval_bs"]
    enc.eval(); head.eval(); res = {}
    for b, ids, att, xn, xc, _ in batches(P, idx, bs, False):
        with torch.autocast("cuda", dtype=AMP, enabled=dev == "cuda"):
            h = enc(ids.to(dev), att.to(dev))
        xn, xc = xn.to(dev), xc.to(dev)
        if all_products:
            pr = torch.sigmoid(head.all_logits(h, xn, xc)).cpu().numpy()
            for r, j in enumerate(b): res[j] = pr[r]
        else:
            for r, j in enumerate(b): res[j] = float(torch.sigmoid(head(h, xn, xc)[r]))
    return res


# ============================================================================ WORKER
def worker(mode):
    seed_all(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P = torch.load(f"{OUT}/prep.pt", weights_only=False)
    R = {"mode": mode, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"}
    tr = np.nonzero(P["split"] == "train")[0]; va = np.nonzero(P["split"] == "val")[0]; te = np.nonzero(P["split"] == "test")[0]

    if mode == "seq" and CFG["frozen_baseline"]:  # frozen (no-finetune) embeddings first, for the "pretrained only" baseline
        t = time.time(); enc0 = SeqEncoder(P, lora=False, device=dev)
        np.save(f"{OUT}/emb_frozen.npy", embed_all(enc0, P, dev)); R["frozen_embed_sec"] = round(time.time() - t, 1)
        del enc0; torch.cuda.empty_cache() if dev == "cuda" else None

    enc = SeqEncoder(P, lora=True, device=dev)
    head = Head(enc.hidden, mode, P["Xn"].shape[1], [len(P["cats"][c]) for c in CAT_COLS]).to(dev)
    lora_p = [p for p in enc.backbone.parameters() if p.requires_grad]
    groups = [{"params": lora_p, "lr": CFG["lr_lora"], "weight_decay": 0.0},
              {"params": list(enc.special.parameters()), "lr": CFG["lr_head"] / 2, "weight_decay": 0.0}]
    if mode == "joint":
        dcn_p = list(head.dcn.parameters()); other = [p for n, p in head.named_parameters() if not n.startswith("dcn.")]
        groups += [{"params": dcn_p, "lr": CFG["lr_head"], "weight_decay": CFG["wd_dcn"]}, {"params": other, "lr": CFG["lr_head"], "weight_decay": 0.0}]
    else:
        groups += [{"params": list(head.parameters()), "lr": CFG["lr_head"], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups)
    R["trainable_params_M"] = round(sum(p.numel() for g in groups for p in g["params"]) / 1e6, 2)
    steps_total = math.ceil(len(tr) / (CFG["batch"] * CFG["accum"])) * CFG["epochs"]
    base_lrs = [g["lr"] for g in opt.param_groups]
    def set_lr(step):
        w = max(1, int(0.05 * steps_total)); f = step / w if step < w else 0.5 * (1 + math.cos(math.pi * (step - w) / max(1, steps_total - w)))
        for g, b in zip(opt.param_groups, base_lrs): g["lr"] = b * f
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda" and AMP == torch.float16)
    eval_every = max(1, steps_total // (CFG["evals_per_epoch"] * CFG["epochs"]))
    rng = np.random.default_rng(SEED); step, micro, best, losses, curve = 0, 0, -1, [], []
    t0 = time.time(); deadline = t0 + CFG["max_train_hours"] * 3600; stop = False
    log(mode, "steps_total", steps_total, "trainable M", R["trainable_params_M"])
    for ep in range(CFG["epochs"]):
        for b, ids, att, xn, xc, y in batches(P, tr, CFG["batch"], True, rng):
            enc.train(); head.train()
            with torch.autocast("cuda", dtype=AMP, enabled=dev == "cuda"):
                h = enc(ids.to(dev), att.to(dev))
            logit = head(h, xn.to(dev), xc.to(dev))
            loss = F.binary_cross_entropy_with_logits(logit, y.to(dev)) / CFG["accum"]
            scaler.scale(loss).backward(); losses.append(loss.item() * CFG["accum"]); micro += 1
            if micro % CFG["accum"]: continue
            set_lr(step); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
            if step % 10 == 0:
                curve.append((step, round(float(np.mean(losses[-10 * CFG["accum"]:])), 4)))
                log(mode, f"step {step}/{steps_total} loss {curve[-1][1]} {(time.time() - t0) / step:.1f}s/step")
            if step % eval_every == 0 or step == steps_total or time.time() > deadline:
                s = score(enc, head, P, va, dev); a = auc(P["y"][va], [s[j] for j in va])
                R.setdefault("val_auc_curve", []).append((step, round(a, 4))); log(mode, "val AUC", round(a, 4))
                if a > best:
                    best = a
                    torch.save({"lora": {k: v for k, v in enc.backbone.state_dict().items() if "lora" in k},
                                "special": enc.special.state_dict(), "head": head.state_dict()}, f"{OUT}/best_{mode}.pt")
                if time.time() > deadline: stop = True; R["stopped_early_for_time"] = step; break
        if stop: break
    R.update(train_sec=round(time.time() - t0, 1), steps=step, best_val_auc=round(best, 4), loss_curve=curve,
             peak_mem_GB=round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev == "cuda" else None)
    # restore best and produce outputs
    ck = torch.load(f"{OUT}/best_{mode}.pt", weights_only=False)
    enc.backbone.load_state_dict(ck["lora"], strict=False); enc.special.load_state_dict(ck["special"]); head.load_state_dict(ck["head"])
    t = time.time()
    if mode == "seq":   # embeddings for late fusion; test scores come from the same pass
        E = embed_all(enc, P, dev); np.save(f"{OUT}/emb_{mode}.npy", E); head.eval()
        with torch.no_grad():
            allp = torch.sigmoid(head.all_logits(torch.from_numpy(E[te]).to(dev), torch.from_numpy(P["Xn"][te]).to(dev),
                                                 torch.from_numpy(P["Xc"][te]).to(dev))).cpu().numpy()
    else:
        s = score(enc, head, P, te, dev, all_products=True)
        allp = np.stack([s[j] for j in te])
    np.save(f"{OUT}/scores_{mode}_test_allproducts.npy", allp)
    R["infer_sec"] = round(time.time() - t, 1)
    json.dump(R, open(f"{OUT}/worker_{mode}.json", "w"), indent=2)
    log(mode, "done", {k: v for k, v in R.items() if k not in ("loss_curve",)})


# ============================================================================ REPORT
def ranking(P, te, allp):
    """Compare model's per-user product ranking to the oracle propensities (only possible with synthetic data)."""
    truth = P["truth"].pivot(index="user_id", columns="product", values="p_true")[PRODUCTS]
    T = truth.loc[P["user_id"][te]].to_numpy()
    best = T.argmax(1); pred = np.argsort(-allp, 1)
    hit1 = float((pred[:, 0] == best).mean()); hit2 = float((pred[:, :2] == best[:, None]).any(1).mean())
    gains = np.take_along_axis(T, pred, 1); ideal = -np.sort(-T, 1); disc = 1 / np.log2(np.arange(2, 7))
    ndcg3 = float(((gains[:, :3] * disc[:3]).sum(1) / (ideal[:, :3] * disc[:3]).sum(1)).mean())
    from scipy.stats import spearmanr
    rho = float(np.nanmean([spearmanr(a, b).statistic for a, b in zip(allp, T)]))
    return {"hit@1": round(hit1, 4), "hit@2": round(hit2, 4), "ndcg@3": round(ndcg3, 4), "spearman": round(rho, 4)}


def report(prep_stats):
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score
    from sklearn.decomposition import PCA
    P = torch.load(f"{OUT}/prep.pt", weights_only=False)
    users = P["users"]; y = P["y"]
    tr, va, te = (np.nonzero(P["split"] == s)[0] for s in ("train", "val", "test"))
    feat = [c for c in users.columns if c not in DROP]
    X = users[feat].copy()
    for c in CAT_COLS: X[c] = pd.Categorical(X[c], categories=P["cats"][c])
    R = {"config": CFG, "data": prep_stats, "rows": {"train": len(tr), "val": len(va), "test": len(te)},
         "label_rate_test": round(float(y[te].mean()), 4), "models": {}}

    def lgbm(Xf, name):
        m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8,
                               subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=int(os.environ.get("LGBM_JOBS", "-1")))
        m.fit(Xf.iloc[tr], y[tr], eval_set=[(Xf.iloc[va], y[va])], callbacks=[lgb.early_stopping(150, verbose=False)])
        p = m.predict_proba(Xf.iloc[te])[:, 1]
        allp = []
        for k in range(len(PRODUCTS)):
            Xk = Xf.iloc[te].copy(); Xk["offered_product"] = pd.Categorical([PRODUCTS[k]] * len(te), categories=PRODUCTS)
            allp.append(m.predict_proba(Xk)[:, 1])
        R["models"][name] = {"test_auc": round(auc(y[te], p), 4), "test_pr_auc": round(float(average_precision_score(y[te], p)), 4),
                             "ranking": ranking(P, te, np.stack(allp, 1)), "trees": int(m.best_iteration_ or m.n_estimators)}
        log(name, R["models"][name])

    R["models"]["oracle (true propensity)"] = {"test_auc": round(auc(y[te], P["p_true_offer"][te]), 4)}
    lgbm(X, "LightGBM tabular (baseline)")

    def with_emb(path, name):
        if not os.path.exists(path): R["models"][name] = {"missing": path}; return
        E = np.load(path); pca = PCA(32, random_state=SEED).fit(E[tr])
        Z = pd.DataFrame(pca.transform(E), columns=[f"e{i}" for i in range(32)], index=X.index)
        lgbm(pd.concat([X, Z], axis=1), name)

    with_emb(f"{OUT}/emb_frozen.npy", "Frozen Qwen3 emb + tabular -> LightGBM (no fine-tune)")
    with_emb(f"{OUT}/emb_seq.npy", "Late fusion: LoRA-seq emb + tabular -> LightGBM")
    for mode, name in (("seq", "LoRA Qwen3 sequence + offer (no tabular)"), ("joint", "nuFormer joint fusion (LoRA Qwen3 + DCNv2/PLR)")):
        f = f"{OUT}/scores_{mode}_test_allproducts.npy"
        if not os.path.exists(f): R["models"][name] = {"missing": f}; continue
        allp = np.load(f); off = P["Xc"][te, 2]; p = allp[np.arange(len(te)), off]
        W = json.load(open(f"{OUT}/worker_{mode}.json"))
        R["models"][name] = {"test_auc": round(auc(y[te], p), 4), "test_pr_auc": round(float(average_precision_score(y[te], p)), 4),
                             "ranking": ranking(P, te, allp), "best_val_auc": W["best_val_auc"], "train_min": round(W["train_sec"] / 60, 1),
                             "steps": W["steps"], "trainable_params_M": W["trainable_params_M"], "peak_mem_GB": W.get("peak_mem_GB")}
        R["worker_" + mode] = W
    # DCNv2 tabular-only (no sequence) — quick, CPU/GPU
    R["models"]["DCNv2+PLR tabular only"] = dcn_only(P, tr, va, te)
    base = R["models"]["LightGBM tabular (baseline)"]["test_auc"]
    for k, v in R["models"].items():
        if "test_auc" in v: v["rel_auc_lift_vs_lgbm_%"] = round(100 * (v["test_auc"] - base) / base, 2)
    json.dump(R, open(f"{OUT}/results.json", "w"), indent=2, default=str)
    plot(R)
    log("REPORT", json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("test_auc", "rel_auc_lift_vs_lgbm_%", "ranking")}
                              for k, v in R["models"].items()}, indent=1))


def dcn_only(P, tr, va, te):
    seed_all(SEED); dev = "cuda" if torch.cuda.is_available() else "cpu"
    dcn = DCNv2(P["Xn"].shape[1], [len(P["cats"][c]) for c in CAT_COLS]).to(dev); lin = nn.Linear(64, 1).to(dev)
    opt = torch.optim.AdamW(list(dcn.parameters()) + list(lin.parameters()), lr=1e-3, weight_decay=1e-4)
    Xn, Xc, y = (torch.from_numpy(a).to(dev) for a in (P["Xn"], P["Xc"], P["y"]))
    f = lambda i: lin(dcn(Xn[i], Xc[i])).squeeze(1)
    best, best_state, rng = -1, None, np.random.default_rng(SEED)
    for ep in range(30):
        dcn.train(); perm = rng.permutation(tr)
        for i in range(0, len(perm), 256):
            b = torch.from_numpy(perm[i:i + 256]).to(dev)
            loss = F.binary_cross_entropy_with_logits(f(b), y[b]); opt.zero_grad(); loss.backward(); opt.step()
        dcn.eval()
        with torch.no_grad(): a = auc(P["y"][va], f(torch.from_numpy(va).to(dev)).cpu().numpy())
        if a > best: best, best_state = a, ({k: v.clone() for k, v in dcn.state_dict().items()}, {k: v.clone() for k, v in lin.state_dict().items()})
    dcn.load_state_dict(best_state[0]); lin.load_state_dict(best_state[1]); dcn.eval()
    with torch.no_grad():
        ti = torch.from_numpy(te).to(dev); p = torch.sigmoid(f(ti)).cpu().numpy(); allp = []
        for k in range(len(PRODUCTS)):
            xk = Xc[ti].clone(); xk[:, 2] = k; allp.append(torch.sigmoid(lin(dcn(Xn[ti], xk)).squeeze(1)).cpu().numpy())
    return {"test_auc": round(auc(P["y"][te], p), 4), "ranking": ranking(P, te, np.stack(allp, 1)), "best_val_auc": round(best, 4)}


def plot(R):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ms = [(k, v) for k, v in R["models"].items() if "test_auc" in v]
    ax[0].barh([k for k, _ in ms], [v["test_auc"] for _, v in ms]); ax[0].set_xlim(0.5, 1); ax[0].set_title("Test AUC")
    for mode in ("seq", "joint"):
        w = R.get("worker_" + mode)
        if w and w.get("loss_curve"): s, l = zip(*w["loss_curve"]); ax[1].plot(s, l, label=mode)
    ax[1].set_title("Train loss"); ax[1].legend(); plt.tight_layout(); plt.savefig(f"{OUT}/results.png", dpi=110)


# ============================================================================ MAIN
if __name__ == "__main__":
    if ARGS.worker:
        worker(ARGS.worker); sys.exit(0)
    T0 = time.time()
    if not SMOKE:
        assert torch.cuda.is_available(), "No GPU attached (Kaggle: phone-verify account / check GPU quota)"
        log("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    os.makedirs(OUT, exist_ok=True)
    stats = prep()
    n_gpu = torch.cuda.device_count()
    py = [sys.executable, os.path.abspath(__file__)] + (["--smoke"] if SMOKE else [])
    modes = [m for m in CFG["modes"].split(",") if m]
    if n_gpu >= 2 and len(modes) > 1:   # run fine-tunes in parallel, one per GPU
        procs = [subprocess.Popen(py + ["--worker", m, "--gpu", str(g)]) for g, m in enumerate(modes)]
        rc = [p.wait() for p in procs]
    else:
        rc = [subprocess.run(py + ["--worker", m, "--gpu", "0"]).returncode for m in modes]
    log("worker return codes", rc)
    report(stats)
    log("total minutes", round((time.time() - T0) / 60, 1))
