"""Local web page: paste a transcript, get the sectioned note with every clause checked against it.

usage: python -m groundnote.app MODEL [--port 8000] [--host 127.0.0.1]   then open http://localhost:8000
MODEL is a local dir or a Hugging Face id (Opflare1144/groundnote-239m). Standard library only; by default it listens
on 127.0.0.1, so nothing leaves the machine.
"""
import argparse
import html
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

from groundnote.check import parse_dialogue
from groundnote.scribe import load, render_html, write_note

MAX_BYTES = 30_000                                           # a long visit is ~15 KB
MAX_TURNS = 300                                              # the longest benchmark visit has 136
LOCK = threading.Lock()                                      # one model on one CPU: one note at a time, others get 503
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
        "/ <code>Patient: ...</code>. Writing the note takes from a few seconds up to a couple of minutes. Research prototype, not for clinical "
        "use.</p><p><b>Do not paste real patient information.</b> On the hosted demo your text is sent to a Hugging Face "
        "server; use the sample or an invented visit.</p><form method=post><textarea name=t rows=22 style='width:100%'>{}"
        "</textarea><p>{}<button>Write note</button></p></form>")


def handler(model, tok):
    class Handler(BaseHTTPRequestHandler):
        timeout = 10                                         # socket timeout: drop clients that stall (slowloris)

        def reply(self, body, code=200):
            data = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(FORM.format(html.escape(SAMPLE), ""))

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if n < 0:
                return self.reply(FORM.format("", "<b>Bad request.</b> "), 400)
            if n > MAX_BYTES:
                return self.reply(FORM.format("", "<b>Transcript too long.</b> "), 413)
            text = parse_qs(self.rfile.read(n).decode("utf-8", "replace")).get("t", [""])[0]
            turns = parse_dialogue(text)
            if not turns:
                return self.reply(FORM.format(html.escape(text), "<b>No 'Doctor:' / 'Patient:' lines found.</b> "), 400)
            if len(turns) > MAX_TURNS:
                return self.reply(FORM.format(html.escape(text), f"<b>Too many turns (over {MAX_TURNS}).</b> "), 413)
            if not LOCK.acquire(blocking=False):
                return self.reply(FORM.format(html.escape(text), "<b>Busy, try again in a minute.</b> "), 503)
            try:
                sections = write_note(model, tok, "\n".join(f"{s.capitalize()}: {u}" for s, u in turns))
                page = render_html(sections, turns)
            except Exception:
                traceback.print_exc()                        # to the server log only, never to the client
                return self.reply(FORM.format(html.escape(text), "<b>Something went wrong. Please try again.</b> "), 500)
            finally:
                LOCK.release()
            self.reply(page)

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")         # 0.0.0.0 only inside a container (the HF Space)
    a = ap.parse_args()
    model, tok = load(a.model_dir)
    print(f"open http://localhost:{a.port}")
    ThreadingHTTPServer((a.host, a.port), handler(model, tok)).serve_forever()   # LOCK serialises the model


if __name__ == "__main__":
    main()
