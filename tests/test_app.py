import http.client
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from groundnote import app
from groundnote.app import LOCK, handler
from scripts.train import build_tokenizer

DIALOGUE = "Doctor: Has your knee locked?\nPatient: No, it doesn't.\nDoctor: Any allergies?\nPatient: Penicillin."


def test_form_and_note_page(tmp_path, monkeypatch):
    rows = [{"dialogue": DIALOGUE, "sections": [("PLAN", "Knee brace; no locking.")]}] * 3
    tok = build_tokenizer(rows, str(tmp_path), init=None)
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=4096, pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id, bos_token_id=tok.eos_token_id)).eval()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler(model, tok))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/"
    try:
        r = urllib.request.urlopen(url)
        assert "Has your knee locked" in r.read().decode()                             # sample prefilled
        assert r.headers["Content-Security-Policy"] == "default-src 'none'; style-src 'unsafe-inline'"
        page = urllib.request.urlopen(url, urllib.parse.urlencode({"t": DIALOGUE}).encode()).read().decode()
        assert "<h3>CHIEF COMPLAINT</h3>" in page and "Has your knee locked?" in page and "Not for clinical use" in page
        try:
            urllib.request.urlopen(url, urllib.parse.urlencode({"t": "<script>x</script>"}).encode())
            assert False, "no-turns input must be rejected"
        except urllib.error.HTTPError as e:
            assert e.code == 400 and "&lt;script&gt;" in e.read().decode()   # echoed back escaped
        for bad in ["abc", "-1"]:                                   # no body: the server must not wait for one
            c = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=5)
            c.putrequest("POST", "/")
            c.putheader("Content-Length", bad)
            c.endheaders()
            assert c.getresponse().status == 400
            c.close()
        with LOCK:                                                  # a note already in progress
            try:
                urllib.request.urlopen(url, urllib.parse.urlencode({"t": DIALOGUE}).encode())
                assert False, "a second note must get 503 while one is running"
            except urllib.error.HTTPError as e:
                assert e.code == 503

        def boom(*a, **k):
            raise RuntimeError("secret detail")
        monkeypatch.setattr(app, "write_note", boom)
        try:
            urllib.request.urlopen(url, urllib.parse.urlencode({"t": DIALOGUE}).encode())
            assert False, "a failing note must get 500"
        except urllib.error.HTTPError as e:
            assert e.code == 500 and "secret detail" not in e.read().decode()   # no traceback to the client
        assert LOCK.acquire(blocking=False)                         # and the lock was released
        LOCK.release()
    finally:
        srv.shutdown()
