"""Gate for the checker on human gold notes (ACI-Bench train+valid; test splits stay untouched).
false alarm = gold clause graded 'contradicted' (gold is faithful).
flip recall = a supported gold clause with its negation flipped on purpose, graded 'contradicted'."""
import csv, re, sys
from collections import Counter, defaultdict
from groundnote.check import check_note, parse_dialogue

NEG_CUE = re.compile(r"\b(no|not|denies|denied|without)\b\s*", re.I)


def flip(text):
    return NEG_CUE.sub("", text, count=1) if NEG_CUE.search(text) else "Denies " + text[0].lower() + text[1:]


stats = defaultdict(Counter)
for f in sys.argv[1:]:
    for r in csv.DictReader(open(f, encoding="utf-8")):
        turns, ds = parse_dialogue(r["dialogue"]), r["dataset"]
        for c in check_note(r["note"], turns):
            stats[ds][c.verdict] += 1
            if c.verdict == "supported":
                stats[ds]["flip_n"] += 1
                stats[ds]["flip_hit"] += [x.verdict for x in check_note(flip(c.text), turns)] == ["contradicted"]
for ds, s in sorted(stats.items()):
    n = s["supported"] + s["unsupported"] + s["contradicted"] + s["unclear"]
    print(f"{ds:11s} clauses={n:4d} supported={s['supported'] / n:.1%} unsupported={s['unsupported'] / n:.1%} "
          f"FALSE_ALARM={s['contradicted'] / n:.1%} flip_recall={s['flip_hit'] / max(1, s['flip_n']):.1%}")
