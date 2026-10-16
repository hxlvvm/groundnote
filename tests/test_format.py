from scripts.train import build_tokenizer, encode

ROW = {"dialogue": "Doctor: start of visit.\n" + "Patient: filler words here.\n" * 300 + "Doctor: end of visit.",
       "sections": [("CHIEF COMPLAINT", "Knee pain."), ("PLAN", "Ice and rest.")]}


def test_encode_masks_dialogue_and_truncates_middle(tmp_path):
    tok = build_tokenizer([ROW], str(tmp_path), init=None)
    ex = encode(tok, ROW, max_len=256)
    assert len(ex["input_ids"]) <= 256 and len(ex["input_ids"]) == len(ex["labels"])
    target = [t for t in ex["labels"] if t != -100]
    assert tok.decode(target).startswith("<section> CHIEF COMPLAINT") and target[-1] == tok.eos_token_id
    text = tok.decode(ex["input_ids"])
    assert "start of visit" in text and "end of visit" in text


def test_positive_fact_moves_negation_into_status():
    from scripts.prep_data import positive_fact
    assert positive_fact({"fact": "no recent travel", "status": "absent"}) == {"fact": "recent travel", "status": "absent"}
    assert positive_fact({"fact": "No fever", "status": "present"}) == {"fact": "fever", "status": "absent"}
    assert positive_fact({"fact": "knee pain", "status": "present"}) == {"fact": "knee pain", "status": "present"}
