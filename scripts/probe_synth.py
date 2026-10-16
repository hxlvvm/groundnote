"""Gate on a synthetic batch: completeness, negatives share, duplication, and whether each gold fact is findable in
the dialogue with its gold polarity (checker as a rough instrument; low agreement = teacher dialogue skips facts)."""
import json, sys
from collections import Counter
from groundnote.check import check_note

recs = [json.loads(l) for l in open(sys.argv[1])]
if not recs:
    sys.exit("no valid records")
v, status, firsts, turns_n = Counter(), Counter(), Counter(), []
for r in recs:
    turns = [(t["speaker"], t["text"]) for t in r["dialogue"]]
    turns_n.append(len(turns))
    firsts[r["dialogue"][1]["text"][:60].lower()] += 1
    for fct in r["facts"]:
        status[fct["status"]] += 1
        claim = fct["fact"] if fct["status"] != "absent" else "Denies " + fct["fact"]
        got = [c.verdict for c in check_note(claim, turns)]
        v[(fct["status"], got[0] if got else "empty")] += 1
bigrams = [tuple(w[i:i + 2]) for r in recs for w in [" ".join(t["text"] for t in r["dialogue"]).lower().split()]
           for i in range(len(w) - 1)]
print(f"distinct-2 over dialogues={len(set(bigrams)) / len(bigrams):.3f}")
empty = sum(1 for r in recs for s, t in r["note"].items() if not t.strip())
print(f"records={len(recs)} turns/rec={sum(turns_n) / len(recs):.1f} facts={sum(status.values())} status={dict(status)}")
print(f"empty note sections={empty} duplicate 2nd turns={sum(c - 1 for c in firsts.values())}")
for k in sorted(v):
    print(" ", k, v[k])
r = recs[0]
print("\nEXAMPLE facts:", r["facts"][:5], "\nDIALOGUE:", *[f"{t['speaker']}: {t['text']}" for t in r["dialogue"][:12]], sep="\n")
print("NOTE:", json.dumps(r["note"], indent=1)[:1500])
