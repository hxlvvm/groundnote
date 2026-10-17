"""Pretrain a ~240M Llama from scratch on FineWeb-Edu, for `scripts.train --init RUN/final`.

usage: python -m scripts.pretrain tok   PT_DIR DATA_DIR     BPE 32k on ~300 MB FineWeb + train.jsonl -> PT_DIR/tok
       python -m scripts.pretrain pack  PT_DIR DATA_DIR     all four shards -> PT_DIR/tokens.bin + val.bin (uint16)
       python -m scripts.pretrain train PT_DIR RUN_DIR STOP_AT   stops cleanly at epoch-seconds STOP_AT; rerun resumes; RUN_DIR/final at the end
"""
import glob
import os
import sys
from itertools import chain

import numpy as np
import pyarrow.parquet as pq
import torch
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from transformers import (LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, Trainer,
                          TrainingArguments, default_data_collator, set_seed)
from transformers.trainer_utils import get_last_checkpoint

from scripts.train import SPECIAL, StopAfter, load, target_text

FINEWEB = "fineweb/sample/10BT/00[0-3]_00000.parquet"
CTX, BATCH, ACCUM = 3072, 8, 16                              # 8 x 16 = 128 seqs ~ 0.4M tokens/step
VAL_TOKENS = 2_000_000


def fineweb_files(data_dir):
    files = sorted(glob.glob(f"{data_dir}/{FINEWEB}"))
    assert len(files) == 4, files                            # download may still be running
    return files


def docs(files, max_chars=None):
    """Yield lists of FineWeb texts, stopping after ~max_chars."""
    n = 0
    for f in files:
        for b in pq.ParquetFile(f).iter_batches(batch_size=4096, columns=["text"]):
            texts = b.column(0).to_pylist()
            yield texts
            n += sum(map(len, texts))
            if max_chars and n >= max_chars:
                return


def build_tok(texts, out):
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(texts, vocab_size=32768, min_frequency=2, special_tokens=SPECIAL)
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe._tokenizer, pad_token="<pad>",
                                  additional_special_tokens=["<dialogue>", "<section>", "<eos>"])
    tok.eos_token = "<eos>"
    tok.save_pretrained(out)
    return tok


def pack_docs(batches, pt_dir, val_tokens=VAL_TOKENS):
    """Encode each doc + <eos> into one flat uint16 file; move the last val_tokens to val.bin."""
    tk = Tokenizer.from_file(f"{pt_dir}/tok/tokenizer.json")
    eos = tk.token_to_id("<eos>")
    assert tk.get_vocab_size() <= 65536
    path = f"{pt_dir}/tokens.bin"
    with open(path, "wb") as f:
        for texts in batches:                                # Rust encode_batch runs on all cores
            enc = tk.encode_batch_fast(texts, add_special_tokens=False)
            np.fromiter(chain.from_iterable(e.ids + [eos] for e in enc), np.uint16).tofile(f)
    m = np.memmap(path, np.uint16, mode="r")
    n = len(m) - val_tokens
    assert n > 0, len(m)
    m[n:].tofile(f"{pt_dir}/val.bin")
    del m
    os.truncate(path, n * 2)
    print(f"train tokens={n} val tokens={val_tokens}")


class Slices(torch.utils.data.Dataset):
    """Fixed CTX-token slices of a uint16 memmap; labels = input_ids (the model shifts)."""

    def __init__(self, path):
        self.path, self.n = path, os.path.getsize(path) // 2 // CTX

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        m = np.memmap(self.path, np.uint16, mode="r")       # opened per item: a memmap pickles as a full copy
        x = torch.from_numpy(m[i * CTX:(i + 1) * CTX].astype(np.int64))
        return {"input_ids": x, "labels": x}


def tok(pt_dir, data_dir):
    task = (r["dialogue"] + "\n" + target_text(r["sections"]) for r in load(f"{data_dir}/train.jsonl"))
    t = build_tok(chain(chain.from_iterable(docs(fineweb_files(data_dir)[:1], 300_000_000)), task), f"{pt_dir}/tok")
    print("vocab", len(t))


def pack(pt_dir, data_dir):
    pack_docs(docs(fineweb_files(data_dir)), pt_dir)


def train(pt_dir, run_dir, stop_at):
    set_seed(0)
    tk = PreTrainedTokenizerFast.from_pretrained(f"{pt_dir}/tok")
    ds = Slices(f"{pt_dir}/tokens.bin")
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(tk), hidden_size=1024, intermediate_size=2816, num_hidden_layers=16, num_attention_heads=16,
        num_key_value_heads=16, max_position_embeddings=CTX, tie_word_embeddings=True, rope_theta=10000.0,
        pad_token_id=tk.pad_token_id, eos_token_id=tk.eos_token_id, bos_token_id=tk.eos_token_id,
        attn_implementation="sdpa"))
    n = sum(p.numel() for p in model.parameters())
    print(f"params={n / 1e6:.1f}M vocab={len(tk)} seqs={len(ds)}")
    assert n > 175e6
    args = TrainingArguments(
        output_dir=run_dir, max_steps=len(ds) // (BATCH * ACCUM), learning_rate=6e-4, lr_scheduler_type="cosine",
        warmup_steps=0.02, weight_decay=0.1, adam_beta2=0.95, max_grad_norm=1.0, per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH, gradient_accumulation_steps=ACCUM, bf16=True,
        save_steps=500, save_total_limit=2,                  # ~20 min on H100, ~50 on A100 (estimated, not measured)
        logging_steps=50, report_to="none", seed=0, dataloader_num_workers=2)
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=default_data_collator,
                      callbacks=[StopAfter(float(stop_at))])
    os.makedirs(run_dir, exist_ok=True)
    trainer.train(resume_from_checkpoint=get_last_checkpoint(run_dir))
    if trainer.state.global_step < args.max_steps:
        print(f"stopped at step {trainer.state.global_step}/{args.max_steps}; rerun to resume")
        return
    trainer.save_model(f"{run_dir}/final")
    tk.save_pretrained(f"{run_dir}/final")
    print("val", trainer.evaluate(Slices(f"{pt_dir}/val.bin")))


if __name__ == "__main__":
    {"tok": tok, "pack": pack, "train": train}[sys.argv[1]](*sys.argv[2:])
