"""Document frequency of content stems over TRAIN dialogues (ACI-Bench train+valid, MTS-Dialog train) -> groundnote/df.json.
The checker uses it to pick the specific term of a claim ("lock") over the generic one ("knee")."""
import csv, json, sys
from collections import Counter
from groundnote.check import content_stems

data = sys.argv[1]
docs = [r["dialogue"] for f in ("aci-bench/data/challenge_data/train.csv", "aci-bench/data/challenge_data/valid.csv",
                                 "MTS-Dialog/Main-Dataset/MTS-Dialog-TrainingSet.csv")
        for r in csv.DictReader(open(f"{data}/{f}", encoding="utf-8"))]
df = Counter(w for d in docs for w in content_stems(d))
json.dump({"n_docs": len(docs), "df": dict(df.most_common())}, open("groundnote/df.json", "w"))
print(len(docs), "docs", len(df), "stems", {w: df[w] for w in ("kne", "lock", "right", "fev", "chill", "swel", "smok")})
