import torch
from transformers import LlamaConfig, LlamaForCausalLM

from groundnote.check import parse_dialogue
from groundnote.scribe import flags, write_note
from scripts.train import build_tokenizer

DIALOGUE = ("Doctor: What brings you in today?\nPatient: My right knee hurts when I climb stairs.\n"
            "Doctor: Any swelling?\nPatient: Yes, some swelling after soccer.\nDoctor: Any allergies?\nPatient: No.")
HEADERS = ["CHIEF COMPLAINT", "ALLERGIES", "PLAN"]


@torch.no_grad()
def uncached_write_note(model, tok, dialogue, headers, n_best=1, max_len=3072, max_new=400):
    """The pre-cache write_note, verbatim: re-prefills the whole prompt for every section."""
    stop = [tok.convert_tokens_to_ids("<section>"), tok.eos_token_id]
    ids = tok("<dialogue>\n" + dialogue + "\n", add_special_tokens=False)["input_ids"]
    budget = max_len - max_new * 3
    if len(ids) > budget:
        ids = ids[: budget * 3 // 5] + ids[-(budget - budget * 3 // 5):]
    turns, out = parse_dialogue(dialogue), []
    for h in headers:
        ids = ids + tok(f"<section> {h}\n", add_special_tokens=False)["input_ids"]
        x = torch.tensor([ids], device=model.device)
        kw = dict(do_sample=True, temperature=0.7, top_p=0.9, num_return_sequences=n_best) if n_best > 1 else dict(do_sample=False)
        gen = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=max_new, eos_token_id=stop,
                             pad_token_id=tok.pad_token_id, repetition_penalty=1.1, **kw)[:, x.shape[1]:]
        texts = [tok.decode([t for t in g.tolist() if t not in stop and t != tok.pad_token_id]).strip() for g in gen]
        best = min(range(len(texts)), key=lambda i: flags(texts[i], turns)) if n_best > 1 else 0
        out.append((h, texts[best]))
        ids = ids + tok(texts[best] + "\n", add_special_tokens=False)["input_ids"]
    return out


def test_cached_write_note_matches_uncached(tmp_path, monkeypatch):
    import groundnote.scribe as scribe
    crops, real_crop = [], scribe.DynamicCache.crop
    monkeypatch.setattr(scribe.DynamicCache, "crop", lambda self, n: (crops.append(n), real_crop(self, n))[1])
    rows = [{"dialogue": DIALOGUE, "sections": [(h, "Knee pain after soccer, no allergies.") for h in HEADERS]}] * 3
    tok = build_tokenizer(rows, str(tmp_path), init=None)
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, max_position_embeddings=512, pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id, bos_token_id=tok.eos_token_id)).eval()
    fed = []  # tokens pushed through the model, per write_note call
    model.register_forward_pre_hook(lambda m, a, kw: fed.__setitem__(-1, fed[-1] + kw["input_ids"].numel()),
                                    with_kwargs=True)

    for n_best in (1, 2):
        torch.manual_seed(1)
        fed.append(0)
        ref = uncached_write_note(model, tok, DIALOGUE, HEADERS, n_best=n_best, max_new=16)
        torch.manual_seed(1)
        fed.append(0)
        crops.clear()
        got = write_note(model, tok, DIALOGUE, HEADERS, n_best=n_best, max_new=16)
        assert got == ref and [h for h, _ in got] == HEADERS
        assert fed[-1] < fed[-2], f"cache not reused: fed {fed[-1]} tokens vs {fed[-2]} uncached"
        assert any(n < 0 for n in crops), "re-tokenization crop path never exercised"
    print("tokens fed (uncached, cached) greedy:", fed[:2], "n_best=2:", fed[2:])
