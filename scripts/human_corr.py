"""Independent check of the checker: MTS-Dialog Correlation-Study has human HallucinationRate labels for 400 model
summaries. Correlate our per-summary flag rate (unsupported+contradicted share of clauses) with them."""
import csv, sys
from groundnote.check import check_note, parse_dialogue

d = sys.argv[1] + "/MTS-Dialog/Correlation-Study/"
summ = list(csv.DictReader(open(d + "MTS-Dialog-Automatic-Summaries-ValidationSet.csv", encoding="utf-8-sig")))
hum = list(csv.DictReader(open(d + "MTS-Dialog-Manual-Scores4CorrelationStudy.csv", encoding="utf-8-sig")))
xs, ys = [], []
for s, h in zip(summ, hum):
    cl = check_note(s["Automatic Summary"], parse_dialogue(s["Dialogue"]))
    if cl:
        xs.append(sum(c.verdict in ("unsupported", "contradicted") for c in cl) / len(cl))
        ys.append(float(h["HallucinationRate"]))


def rank(v):
    o = sorted(range(len(v)), key=v.__getitem__)
    r = [0.0] * len(v)
    i = 0
    while i < len(o):                       # average ranks for ties
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2
        i = j + 1
    return r


def pearson(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return num / (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5


print(f"n={len(xs)} spearman={pearson(rank(xs), rank(ys)):.3f} pearson={pearson(xs, ys):.3f}")
hi = [x for x, y in zip(xs, ys) if y > 0]
lo = [x for x, y in zip(xs, ys) if y == 0]
print(f"mean flag rate: human-hallucinated={sum(hi) / len(hi):.3f} (n={len(hi)})  clean={sum(lo) / len(lo):.3f} (n={len(lo)})")
