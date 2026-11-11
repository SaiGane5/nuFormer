"""Chunked generator for large datasets (same rules as generate.py, bounded memory).

python -m synthgen.generate_big --n-users 300000 --chunk 50000 --seed 7 --out data_300k

Each chunk is an independent draw of `chunk` customers with its own seed; user_ids are offset so they stay
unique and ascending. The data-generating process (personas, events, transactions, labels, features) is
exactly the one in generate.py, so results stay comparable with the 20k runs.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import GenConfig
from .generate import __version__, generate
from .validate import validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-users", type=int, required=True)
    ap.add_argument("--chunk", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    base = GenConfig()
    writers, chunk_stats, t0 = {}, [], time.time()
    n_chunks = (a.n_users + a.chunk - 1) // a.chunk
    for i in range(n_chunks):
        n = min(a.chunk, a.n_users - i * a.chunk)
        cfg = replace(base, n_users=n, seed=a.seed + 7919 * i)
        d = generate(cfg)
        off = i * a.chunk
        for k in d:
            d[k]["user_id"] = d[k]["user_id"].astype(np.int64) + off
        chunk_stats.append(validate(d["users"], d["transactions"], d["truth"]))
        for k, fname in (("users", "users"), ("transactions", "transactions"), ("truth", "product_truth"), ("latents", "latents")):
            t = pa.Table.from_pandas(d[k], preserve_index=False)
            if k not in writers:
                writers[k] = pq.ParquetWriter(f"{a.out}/{fname}.parquet", t.schema)
            writers[k].write_table(t.cast(writers[k].schema))
        print(f"chunk {i + 1}/{n_chunks}: {n} users, {len(d['transactions'])} tx, {time.time() - t0:.0f}s", flush=True)
        del d
    for w in writers.values():
        w.close()
    meta = {"generator_version": __version__, "chunked": True, "chunk": a.chunk, "n_chunks": n_chunks,
            "config": asdict(replace(base, n_users=a.n_users, seed=a.seed)), "seconds": round(time.time() - t0, 1),
            "chunk_stats": chunk_stats}
    json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=2, default=str)
    print(json.dumps({k: v for k, v in meta.items() if k != "chunk_stats"}, indent=2, default=str))


if __name__ == "__main__":
    main()
