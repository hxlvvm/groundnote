"""Pre-registered quality gate for a teacher's pilot file. Exit 0 = pass, 1 = fail.

usage: python -m scripts.gate_teacher PILOT.jsonl
Pass rule (fixed 2026-10-15 before the GLM v4 pilot ran; Gemma 31B QAT scored 2.3% / 21.9% / 82%):
  note clauses contradicting the teacher's own dialogue <= 3.0%
  note clauses not found in the dialogue                <= 30%
  numbers in the note that are also spoken               >= 75%
"""
import json
import re
import sys

from groundnote.check import check_note

MAX_CONTRA, MAX_UNSUP, MIN_NUMBERS = 0.03, 0.30, 0.75


def metrics(recs):
    tot = contra = unsup = nums = grounded = 0
    for r in recs:
        turns = [(t["speaker"], t["text"]) for t in r["dialogue"]]
        for c in check_note("\n".join(r["note"].values()), turns):
            tot += 1
            contra += c.verdict == "contradicted"
            unsup += c.verdict == "unsupported"
        dia = " ".join(t["text"] for t in r["dialogue"]).lower()
        for num in set(re.findall(r"\b\d{2,3}(?:/\d{2,3})?\b", " ".join(r["note"].values()))):
            nums += 1
            grounded += num in dia
    return contra / max(1, tot), unsup / max(1, tot), grounded / max(1, nums)


if __name__ == "__main__":
    recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    c, u, g = metrics(recs)
    ok = len(recs) >= 50 and c <= MAX_CONTRA and u <= MAX_UNSUP and g >= MIN_NUMBERS
    print(f"{sys.argv[1]}: n={len(recs)} contradicted={c:.1%} unsupported={u:.1%} numbers_grounded={g:.0%} -> "
          f"{'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
