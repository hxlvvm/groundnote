"""Write a sectioned clinical note from a transcript, then grade every clause against the transcript.

usage: python -m groundnote.scribe MODEL_DIR transcript.txt [--n-best 4] [--html note.html] [--device cpu]
Transcript lines look like "Doctor: ..." / "Patient: ..." (or "[doctor] ...").
"""
import argparse
import html
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from groundnote.check import check_note, parse_dialogue

STANDARD = ["CHIEF COMPLAINT", "HISTORY OF PRESENT ILLNESS", "PAST MEDICAL HISTORY", "ALLERGIES", "PHYSICAL EXAM",
            "ASSESSMENT", "PLAN"]
COLOR = {"supported": "#1a7f37", "contradicted": "#cf222e", "unsupported": "#9a6700", "unclear": "#6e7781"}


def flags(text, turns):
    cl = check_note(text, turns)
    return 2 * sum(c.verdict == "contradicted" for c in cl) + sum(c.verdict == "unsupported" for c in cl)


@torch.no_grad()
def write_note(model, tok, dialogue, headers=STANDARD, n_best=1, max_len=3072, max_new=400):
    """Forced headers, one section at a time; with n_best > 1, keep the sample the checker flags least."""
    stop = [tok.convert_tokens_to_ids("<section>"), tok.eos_token_id]
    ids = tok("<dialogue>\n" + dialogue + "\n", add_special_tokens=False)["input_ids"]
    budget = max_len - max_new * 3
    if len(ids) > budget:                                    # same head+tail truncation as training
        ids = ids[: budget * 3 // 5] + ids[-(budget - budget * 3 // 5):]
    turns, out = parse_dialogue(dialogue), []
    cache, seq = DynamicCache(), []                          # seq: the token ids whose K/V the cache holds
    for h in headers:
        ids = ids + tok(f"<section> {h}\n", add_special_tokens=False)["input_ids"]
        # keep the cache only up to where it agrees with ids (re-tokenized text can differ from the generated ids);
        # leave >= 1 token uncached so generate has something to prefill
        k = next((i for i, (a, b) in enumerate(zip(seq, ids)) if a != b), min(len(seq), len(ids)))
        cache.crop(min(k, len(ids) - 1) - cache.get_seq_length())
        x = torch.tensor([ids], device=model.device)
        kw = dict(do_sample=True, temperature=0.7, top_p=0.9, num_return_sequences=n_best) if n_best > 1 else dict(do_sample=False)
        if n_best > 1:
            cache.batch_repeat_interleave(n_best)            # generate expands input tensors, never a Cache
        # full ids + full mask: generate feeds only the uncached tail, repetition_penalty still sees all of ids
        full = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=max_new, eos_token_id=stop,
                              pad_token_id=tok.pad_token_id, repetition_penalty=1.1, past_key_values=cache, **kw)
        gen = full[:, x.shape[1]:]
        texts = [tok.decode([t for t in g.tolist() if t not in stop and t != tok.pad_token_id]).strip() for g in gen]
        best = min(range(len(texts)), key=lambda i: flags(texts[i], turns)) if n_best > 1 else 0
        if n_best > 1:
            cache.batch_select_indices([best])
        seq = full[best, :cache.get_seq_length()].tolist()
        out.append((h, texts[best]))
        ids = ids + tok(texts[best] + "\n", add_special_tokens=False)["input_ids"]
    return out


def load(model_dir, device="cpu"):
    tok = AutoTokenizer.from_pretrained(model_dir)
    return AutoModelForCausalLM.from_pretrained(model_dir).to(device).eval(), tok


def render_html(sections, turns):
    rows = []
    for h, text in sections:
        rows.append(f"<h3>{html.escape(h)}</h3><p>")
        for c in check_note(text, turns):
            ev = "; ".join(f"[{i + 1}] {turns[i][0]}: {turns[i][1]}" for i in c.evidence if i is not None)
            rows.append(f'<span style="border-bottom:2px solid {COLOR[c.verdict]}" title="{c.verdict}: '
                        f'{html.escape(ev, quote=True)}">{html.escape(c.text)}</span> ')
        rows.append("</p>")
    dia = "".join(f"<li><b>{s}</b>: {html.escape(u)}</li>" for s, u in turns)
    legend = " ".join(f'<span style="color:{v}">&#9632; {k}</span>' for k, v in COLOR.items())
    return (f"<!doctype html><meta charset=utf-8><title>groundnote</title><body style='font:15px system-ui;"
            f"max-width:1100px;margin:auto;padding:16px;display:grid;grid-template-columns:1fr 1fr;gap:24px'>"
            f"<div><h2>Note</h2><p>{legend}<br><small>Hover a clause for its evidence turn.</small></p>{''.join(rows)}</div>"
            f"<div><h2>Transcript</h2><ol>{dia}</ol></div></body>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("transcript")
    ap.add_argument("--n-best", type=int, default=1)
    ap.add_argument("--html")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    torch.manual_seed(0)
    dialogue = open(a.transcript, encoding="utf-8").read()
    turns = parse_dialogue(dialogue)
    if not turns:
        raise SystemExit("no 'Doctor:'/'Patient:' lines found in the transcript")
    model, tok = load(a.model_dir, a.device)
    t0 = time.time()
    sections = write_note(model, tok, "\n".join(f"{s.capitalize()}: {u}" for s, u in turns), n_best=a.n_best)
    print(f"[{time.time() - t0:.1f}s on {a.device}]")
    mark = {"supported": "", "contradicted": "  <<CONTRADICTS TRANSCRIPT", "unsupported": "  <<NOT FOUND IN TRANSCRIPT",
            "unclear": "  <<UNCLEAR"}
    for h, text in sections:
        print(f"\n{h}")
        for c in check_note(text, turns):
            print(f"  {c.text}{mark[c.verdict]}")
    if a.html:
        open(a.html, "w", encoding="utf-8").write(render_html(sections, turns))


if __name__ == "__main__":
    main()
