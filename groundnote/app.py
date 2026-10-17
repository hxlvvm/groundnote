"""Local web page: paste a transcript, get the sectioned note with every clause checked against it.

usage: python -m groundnote.app MODEL [--port 8000] [--host 127.0.0.1]   then open http://localhost:8000
MODEL is a local dir or a Hugging Face id (Opflare1144/groundnote-97m). Standard library only; by default it listens
on 127.0.0.1, so nothing leaves the machine.
"""
import argparse
import html
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs

from groundnote.check import parse_dialogue
from groundnote.scribe import load, render_html, write_note

MAX_BYTES = 200_000                                          # a long visit is ~15 KB
SAMPLE = """Doctor: Hi, what brings you in today?
Patient: My right knee has been hurting since I twisted it playing football two weeks ago.
Doctor: Has your knee locked or given way?
Patient: No, it doesn't lock. It just swells up after I walk a lot.
Doctor: Any numbness or tingling in the leg?
Patient: No.
Doctor: Are you taking anything for the pain?
Patient: Ibuprofen, 400 milligrams, twice a day.
Doctor: Any allergies to medications?
Patient: Penicillin, it gives me a rash.
Doctor: On exam there is mild swelling over the right knee, no instability, full range of motion.
Doctor: This looks like a knee sprain. Rest, ice, and a knee brace, and come back in two weeks if it is not better."""
FORM = ("<!doctype html><meta charset=utf-8><title>groundnote</title><body style='font:15px system-ui;max-width:900px;"
        "margin:auto;padding:16px'><h2>groundnote</h2><p>Paste a transcript, one turn per line: <code>Doctor: ...</code> "
        "/ <code>Patient: ...</code>. Writing the note takes from a few seconds up to a minute. Research prototype, not for clinical "
        "use.</p><form method=post><textarea name=t rows=22 style='width:100%'>{}</textarea><p>{}<button>Write note"
        "</button></p></form>")


def handler(model, tok):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, body, code=200):
            data = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(FORM.format(html.escape(SAMPLE), ""))

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BYTES:
                return self.reply(FORM.format("", "<b>Transcript too long.</b> "), 413)
            text = parse_qs(self.rfile.read(n).decode("utf-8", "replace")).get("t", [""])[0]
            turns = parse_dialogue(text)
            if not turns:
                return self.reply(FORM.format(html.escape(text), "<b>No 'Doctor:' / 'Patient:' lines found.</b> "), 400)
            sections = write_note(model, tok, "\n".join(f"{s.capitalize()}: {u}" for s, u in turns))
            self.reply(render_html(sections, turns))

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")         # 0.0.0.0 only inside a container (the HF Space)
    a = ap.parse_args()
    model, tok = load(a.model_dir)
    print(f"open http://localhost:{a.port}")
    HTTPServer((a.host, a.port), handler(model, tok)).serve_forever()   # one request at a time: one model


if __name__ == "__main__":
    main()
