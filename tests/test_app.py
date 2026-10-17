import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import HTTPServer

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from groundnote.app import handler
from scripts.train import build_tokenizer

DIALOGUE = "Doctor: Has your knee locked?\nPatient: No, it doesn't.\nDoctor: Any allergies?\nPatient: Penicillin."


def test_form_and_note_page(tmp_path):
    rows = [{"dialogue": DIALOGUE, "sections": [("PLAN", "Knee brace; no locking.")]}] * 3
    tok = build_tokenizer(rows, str(tmp_path), init=None)
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=4096, pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id, bos_token_id=tok.eos_token_id)).eval()
    srv = HTTPServer(("127.0.0.1", 0), handler(model, tok))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/"
    try:
        assert "Has your knee locked" in urllib.request.urlopen(url).read().decode()   # sample prefilled
        page = urllib.request.urlopen(url, urllib.parse.urlencode({"t": DIALOGUE}).encode()).read().decode()
        assert "<h3>CHIEF COMPLAINT</h3>" in page and "Has your knee locked?" in page
        try:
            urllib.request.urlopen(url, urllib.parse.urlencode({"t": "<script>x</script>"}).encode())
            assert False, "no-turns input must be rejected"
        except urllib.error.HTTPError as e:
            assert e.code == 400 and "&lt;script&gt;" in e.read().decode()   # echoed back escaped
    finally:
        srv.shutdown()
