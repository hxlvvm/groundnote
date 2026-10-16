"""Independent judge: a different model family (Gemini, free tier) labels every sentence of a note against its
transcript. Used only to EVALUATE notes, never to make training data.

usage: python -m scripts.judge PRED.jsonl OUT.jsonl [--which pred|sections] [--limit N] [--model gemini-3.8-flash]
Key: ~/.config/groundnote/gemini_key (chmod 600) or $GEMINI_API_KEY. Data sent is fictional/role-played, never PHI.
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPT = """You are checking a clinical note against the doctor-patient transcript it was written from.
Split the note into sentences. Label each one:
- SUPPORTED: the transcript states or clearly implies it, with the same polarity (present vs denied/absent).
- CONTRADICTED: the transcript states the opposite, e.g. the note says a symptom is present but the patient denied it.
- UNSUPPORTED: the transcript does not mention it.
Return JSON: {{"sentences": [{{"text": "...", "label": "SUPPORTED|CONTRADICTED|UNSUPPORTED"}}]}}

TRANSCRIPT:
{dialogue}

NOTE:
{note}"""


def key():
    f = Path.home() / ".config/groundnote/gemini_key"
    k = os.environ.get("GEMINI_API_KEY") or (f.read_text().strip() if f.exists() else "")
    if not k:
        raise SystemExit(f"no key: put it in {f} (chmod 600) or set GEMINI_API_KEY")
    return k


def ask(model, k, text, retries=6, temperature=0):
    """One Gemini call returning the parsed JSON answer (None after retries)."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = json.dumps({"contents": [{"parts": [{"text": text}]}],
                       "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"}}).encode()
    for attempt in range(retries):
        req = urllib.request.Request(url, body, {"Content-Type": "application/json", "x-goog-api-key": k})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.load(r)
            return json.loads(out["candidates"][0]["content"]["parts"][0]["text"])
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500:
                raise
            time.sleep(30 * (attempt + 1))                   # free-tier rate limit or transient server error
        except (TimeoutError, urllib.error.URLError):       # after HTTPError (its subclass): network stall, retry
            time.sleep(30 * (attempt + 1))
        except (KeyError, json.JSONDecodeError):
            time.sleep(5)                                    # blocked or malformed answer: retry once more
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred")
    ap.add_argument("out")
    ap.add_argument("--which", default="pred", help="'pred' = model note, 'sections' = gold note (reference rate)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", default="gemini-3.8-flash")   # 2.5 is closed to new keys
    ap.add_argument("--pause", type=float, default=7.0, help="seconds between calls (free tier ~10 requests/min)")
    a = ap.parse_args()
    k = key()
    rows = [json.loads(l) for l in open(a.pred)][: a.limit]
    done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()   # resumable
    with open(a.out, "a") as f:
        for r in rows:
            if r["id"] in done:
                continue
            note = "\n\n".join(f"{h}\n{t}" for h, t in r[a.which])
            ans = ask(a.model, k, PROMPT.format(dialogue=r["dialogue"], note=note))
            labels = ans.get("sentences") if isinstance(ans, dict) else None
            f.write(json.dumps({"id": r["id"], "which": a.which, "model": a.model, "labels": labels}) + "\n")
            f.flush()
            time.sleep(a.pause)
    res = [json.loads(l) for l in open(a.out)]
    lab = [s["label"] for x in res if x["labels"] for s in x["labels"]]
    n = max(1, len(lab))
    print(f"{a.which}: notes={len(res)} sentences={len(lab)} "
          + " ".join(f"{k}={lab.count(k) / n:.1%}" for k in ("SUPPORTED", "CONTRADICTED", "UNSUPPORTED")))


if __name__ == "__main__":
    main()
