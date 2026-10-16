"""Turn a teacher pilot file into judge input: does the teacher's own note match its own dialogue?
usage: python -m scripts.teacher_judge_input PILOT.jsonl OUT.jsonl [N]"""
import json, sys
from scripts.prep_data import render

rows = [json.loads(l) for l in open(sys.argv[1])][: int(sys.argv[3]) if len(sys.argv) > 3 else 15]
with open(sys.argv[2], "w") as f:
    for r in rows:
        f.write(json.dumps({"id": r["id"], "dialogue": render((t["speaker"], t["text"]) for t in r["dialogue"]),
                            "sections": list(r["note"].items())}) + "\n")
print(len(rows), "->", sys.argv[2])
