"""Batch generation over a split: section by section with the gold headers, greedy or best-of-N.

usage: python -m scripts.generate MODEL_DIR SPLIT.jsonl OUT.jsonl [--n-best N] [--limit K] [--device cpu|cuda]
"""
import argparse
import json
import time

import torch

from groundnote.scribe import STANDARD, load, write_note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("split")
    ap.add_argument("out")
    ap.add_argument("--n-best", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.manual_seed(0)
    model, tok = load(a.model_dir, a.device)
    rows = [json.loads(l) for l in open(a.split)][: a.limit]
    with open(a.out, "w") as f:
        for r in rows:
            t0 = time.time()
            pred = write_note(model, tok, r["dialogue"], [h for h, _ in r["sections"]] or STANDARD, a.n_best)
            f.write(json.dumps({**r, "pred": pred, "seconds": round(time.time() - t0, 2)}) + "\n")
    print("wrote", len(rows), a.out)


if __name__ == "__main__":
    main()
