"""Train the note model. Default: tokenizer and weights from scratch. --init HF_ID: same data, pretrained start (ablation).

usage: python -m scripts.train RUN_DIR DATA_DIR [--init HF_ID] [--epochs N] [--lr X] [--max-len N] [--limit N]
Format per example: <dialogue>\\n{dialogue}\\n<section> HEADER\\n{text}\\n ... <eos>; loss on the section part only.
"""
import argparse
import json
import random

import torch
from tokenizers import ByteLevelBPETokenizer
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, LlamaConfig, LlamaForCausalLM,
                          PreTrainedTokenizerFast, Trainer, TrainingArguments)

SPECIAL = ["<pad>", "<eos>", "<dialogue>", "<section>"]


def load(path, limit=None):
    rows = [json.loads(l) for l in open(path)]
    return rows[:limit] if limit else rows


def target_text(sections):
    return "".join(f"<section> {h}\n{t}\n" for h, t in sections) + "<eos>"


def build_tokenizer(rows, run_dir, init):
    if init:
        tok = AutoTokenizer.from_pretrained(init)
        tok.add_special_tokens({"pad_token": "<pad>", "additional_special_tokens": ["<dialogue>", "<section>", "<eos>"]})
    else:
        bpe = ByteLevelBPETokenizer()
        bpe.train_from_iterator((r["dialogue"] + "\n" + target_text(r["sections"]) for r in rows),
                                vocab_size=16384, min_frequency=2, special_tokens=SPECIAL)
        tok = PreTrainedTokenizerFast(tokenizer_object=bpe._tokenizer, pad_token="<pad>",
                                      additional_special_tokens=["<dialogue>", "<section>", "<eos>"])
    tok.eos_token = "<eos>"
    tok.save_pretrained(run_dir)
    return tok


def encode(tok, r, max_len):
    tgt = tok(target_text(r["sections"]), add_special_tokens=False)["input_ids"][: max_len // 2]
    src = tok("<dialogue>\n" + r["dialogue"] + "\n", add_special_tokens=False)["input_ids"]
    budget = max_len - len(tgt)
    if len(src) > budget:                                    # keep the start and the end of a long visit
        src = src[: budget * 3 // 5] + src[-(budget - budget * 3 // 5):]
    return {"input_ids": src + tgt, "labels": [-100] * len(src) + tgt, "length": len(src) + len(tgt)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("data_dir")
    ap.add_argument("--init")
    ap.add_argument("--epochs", type=float, default=6)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)    # big-vocab inits: smaller --batch, same effective batch
    ap.add_argument("--limit", type=int)                     # overfit gate: tiny train set
    a = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    train, dev = load(f"{a.data_dir}/train.jsonl", a.limit), load(f"{a.data_dir}/dev.jsonl")
    if a.limit:
        dev = train                                          # overfit gate measures fit on the same examples
    tok = build_tokenizer(train, a.run_dir, a.init)
    if a.init:
        model = AutoModelForCausalLM.from_pretrained(a.init, dtype=torch.float32)
        model.resize_token_embeddings(len(tok))
    else:
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=len(tok), hidden_size=768, intermediate_size=2048, num_hidden_layers=12, num_attention_heads=12,
            num_key_value_heads=12, max_position_embeddings=a.max_len, tie_word_embeddings=True, rope_theta=10000.0,
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, bos_token_id=tok.eos_token_id))
    model.config.pad_token_id, model.config.eos_token_id = tok.pad_token_id, tok.eos_token_id
    print(f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M vocab={len(tok)} train={len(train)} dev={len(dev)}")
    tr = [encode(tok, r, a.max_len) for r in train]
    dv = [encode(tok, r, a.max_len) for r in dev]
    print("tokens/epoch", sum(x["length"] for x in tr), "target tokens", sum(sum(l != -100 for l in x["labels"]) for x in tr))
    args = TrainingArguments(
        output_dir=a.run_dir, num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_steps=0.03, weight_decay=0.1, per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum,
        bf16=True, train_sampling_strategy="group_by_length", length_column_name="length", eval_strategy="epoch",
        save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        logging_steps=50, report_to="none", seed=0, dataloader_num_workers=2, max_grad_norm=1.0)
    pad = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)
    trainer = Trainer(model=model, args=args, train_dataset=tr, eval_dataset=dv, processing_class=tok,
                      data_collator=lambda b: pad([{k: v for k, v in x.items() if k != "length"} for x in b]))
    trainer.train()
    trainer.save_model(f"{a.run_dir}/final")
    tok.save_pretrained(f"{a.run_dir}/final")
    print("final eval", trainer.evaluate())


if __name__ == "__main__":
    main()
