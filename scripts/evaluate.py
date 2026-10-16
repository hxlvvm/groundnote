"""Score generated notes. usage: python -m scripts.evaluate PRED.jsonl [PRED2.jsonl ...]

- ROUGE-1/2/Lsum on the full note (headers + text), per source; ACI test1 reported alone (published baselines use it).
- Checker flag rates on predictions and, for reference, on the gold notes of the same encounters.
- Synthetic only, against GOLD fact labels (not the dialogue, so independent of the reranker's signal):
  fact recall and the negation-flip rate = absent facts the note asserts as present (the post's "knee locked" bug).
"""
import json
import sys
from collections import defaultdict

from rouge_score import rouge_scorer

from groundnote.check import DF, SUPPORT_THRESHOLD, check_note, clause_polarity, content_stems, parse_dialogue, split_clauses

SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)


def note_text(sections):
    return "\n\n".join(f"{h}\n{t}" for h, t in sections)


def fact_eval(facts, pred_sections):
    clauses = [c for _, t in pred_sections for line in t.splitlines() for c in split_clauses(line)]
    found = flips = absent = 0
    for f in facts:
        if f["status"] == "uncertain":
            continue
        s = {w for w in content_stems(f["fact"]) if not w.isdigit()}
        if not s:
            continue
        absent += f["status"] == "absent"
        best = max(clauses, key=lambda c: len(s & content_stems(c)), default="")
        matched = s & content_stems(best)
        if not s or len(matched) / len(s) < SUPPORT_THRESHOLD:
            continue
        found += 1
        rarest = min(DF.get(w, 0) for w in matched)
        p = clause_polarity(best, {w for w in matched if DF.get(w, 0) == rarest})
        flips += (p == 1 and f["status"] == "absent")
    n = sum(f["status"] != "uncertain" for f in facts)
    return found, flips, n, absent


def main(paths):
    for path in paths:
        rows = [json.loads(l) for l in open(path)]
        groups = defaultdict(list)
        for r in rows:
            src = "aci-test1" if "taskB_test1" in r["id"] else r["source"].split("-")[0]
            groups[src].append(r)
            if src.startswith("aci"):
                groups["aci-all"].append(r)
        print(f"\n### {path}  n={len(rows)}  median sec/note={sorted(r['seconds'] for r in rows)[len(rows) // 2]}")
        for src, rs in sorted(groups.items()):
            rg = [SCORER.score(note_text(r["sections"]), note_text(r["pred"])) for r in rs]
            m = {k: 100 * sum(x[k].fmeasure for x in rg) / len(rg) for k in ("rouge1", "rouge2", "rougeLsum")}
            fl = {"pred": [0, 0, 0], "gold": [0, 0, 0]}
            for r in rs:
                turns = parse_dialogue(r["dialogue"])
                for name, secs in (("pred", r["pred"]), ("gold", r["sections"])):
                    for c in check_note(note_text(secs), turns):
                        fl[name][0] += 1
                        fl[name][1] += c.verdict == "contradicted"
                        fl[name][2] += c.verdict == "unsupported"
            line = (f"{src:10s} n={len(rs):3d} R1={m['rouge1']:.2f} R2={m['rouge2']:.2f} RLsum={m['rougeLsum']:.2f} | "
                    + " ".join(f"{k}: contra={v[1] / max(1, v[0]):.1%} unsup={v[2] / max(1, v[0]):.1%}" for k, v in fl.items()))
            if src == "synthetic":
                tot = [sum(x) for x in zip(*(fact_eval(r["facts"], r["pred"]) for r in rs))]
                line += f" | fact_recall={tot[0] / tot[2]:.1%} absent_asserted={tot[1] / max(1, tot[3]):.1%} ({tot[1]}/{tot[3]})"
            print(line)


if __name__ == "__main__":
    main(sys.argv[1:])
