"""Build train/dev/test examples: one record = dialogue + ordered list of (SECTION, text).
Sources: ACI-Bench (full notes), MTS-Dialog (one section per dialogue, + back-translated train), synthetic encounters.
Test splits of ACI-Bench (3 x 40) and MTS-Dialog (2 x 200) are never trained on.

usage: python -m scripts.prep_data DATA_DIR [SYNTH_GLOB]
"""
import csv
import glob
import json
import random
import re
import sys

from groundnote.check import check_note

ACI_MAP = {"CHIEF COMPLAINT": "CHIEF COMPLAINT", "CC": "CHIEF COMPLAINT",
           "HISTORY OF PRESENT ILLNESS": "HISTORY OF PRESENT ILLNESS", "HPI": "HISTORY OF PRESENT ILLNESS",
           "MEDICAL HISTORY": "PAST MEDICAL HISTORY", "PAST HISTORY": "PAST MEDICAL HISTORY",
           "PAST MEDICAL HISTORY": "PAST MEDICAL HISTORY", "SURGICAL HISTORY": "PAST SURGICAL HISTORY",
           "PAST SURGICAL HISTORY": "PAST SURGICAL HISTORY", "SOCIAL HISTORY": "SOCIAL HISTORY",
           "FAMILY HISTORY": "FAMILY HISTORY", "MEDICATIONS": "MEDICATIONS", "CURRENT MEDICATIONS": "MEDICATIONS",
           "ALLERGIES": "ALLERGIES", "REVIEW OF SYSTEMS": "REVIEW OF SYSTEMS", "REVIEW OF SYMPTOMS": "REVIEW OF SYSTEMS",
           "VITALS": "VITALS", "VITALS REVIEWED": "VITALS", "PHYSICAL EXAM": "PHYSICAL EXAM",
           "PHYSICAL EXAMINATION": "PHYSICAL EXAM", "EXAM": "PHYSICAL EXAM", "RESULTS": "RESULTS",
           "ASSESSMENT": "ASSESSMENT", "IMPRESSION": "ASSESSMENT", "PLAN": "PLAN",
           "ASSESSMENT AND PLAN": "ASSESSMENT AND PLAN", "INSTRUCTIONS": "INSTRUCTIONS"}
MTS_MAP = {"CC": "CHIEF COMPLAINT", "GENHX": "HISTORY OF PRESENT ILLNESS", "PASTMEDICALHX": "PAST MEDICAL HISTORY",
           "PASTSURGICAL": "PAST SURGICAL HISTORY", "FAM/SOCHX": "FAMILY AND SOCIAL HISTORY", "ALLERGY": "ALLERGIES",
           "MEDICATIONS": "MEDICATIONS", "ROS": "REVIEW OF SYSTEMS", "EXAM": "PHYSICAL EXAM", "ASSESSMENT": "ASSESSMENT",
           "DIAGNOSIS": "ASSESSMENT", "PLAN": "PLAN", "DISPOSITION": "PLAN", "IMAGING": "RESULTS", "LABS": "RESULTS"}


NEG_LEAD = re.compile(r"^(no|not|denies|denied|without|absence of|absent|negative for|lack of)\s+(known\s+|recent\s+)?", re.I)


def positive_fact(f):
    """Teachers sometimes write absent facts as 'no recent travel'; gold labels need the polarity in status only."""
    text = NEG_LEAD.sub(lambda m: m.group(2) or "", f["fact"].strip())
    if text != f["fact"].strip() and f["status"] == "present":
        return {**f, "fact": text, "status": "absent"}       # 'no fever' marked present means fever absent
    return {**f, "fact": text}


ACI_REPEAT = 20                                       # oversampling of ACI-Bench train notes (the only real full notes)


def render(turns):
    return "\n".join(f"{s.capitalize()}: {t.strip()}" for s, t in turns)


def aci_turns(text):
    return [(m.group(1), m.group(2)) for m in re.finditer(r"^\[(\w+)\]\s*(.*)$", text, re.M)]


def mts_turns(text):
    out = []
    for line in text.splitlines():
        m = re.match(r"\s*(Doctor|Patient|Guest_family|Guest_clinician)\s*:\s*(.*)", line)
        if m:
            out.append(("doctor" if m.group(1) in ("Doctor", "Guest_clinician") else
                        "family" if m.group(1) == "Guest_family" else "patient", m.group(2)))
        elif line.strip() and out:
            out[-1] = (out[-1][0], out[-1][1] + " " + line.strip())
    return out


def aci_sections(note):
    secs, head = [], None
    for line in note.splitlines():
        h = line.strip().rstrip(":")
        if h.isupper() and h in ACI_MAP:
            head = ACI_MAP[h]
            secs.append([head, []])
        elif line.strip() and head:
            secs[-1][1].append(line.strip())
    return [(h, "\n".join(t)) for h, t in secs if t]


def main(data, synth_glob=None):
    out = {"train": [], "dev": [], "test": []}
    aci = f"{data}/aci-bench/data/challenge_data"
    for f, split in [("train", "train"), ("valid", "dev"), ("clinicalnlp_taskB_test1", "test"),
                     ("clinicalnlp_taskC_test2", "test"), ("clef_taskC_test3", "test")]:
        for r in csv.DictReader(open(f"{aci}/{f}.csv", encoding="utf-8")):
            out[split].append({"id": f"aci-{f}-{r['encounter_id']}", "source": f"aci-{r['dataset']}",
                               "dialogue": render(aci_turns(r["dialogue"])), "sections": aci_sections(r["note"])})
    mts = f"{data}/MTS-Dialog"
    files = [("Main-Dataset/MTS-Dialog-TrainingSet.csv", "train"), ("Main-Dataset/MTS-Dialog-ValidationSet.csv", "dev"),
             ("Main-Dataset/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv", "test"),
             ("Main-Dataset/MTS-Dialog-TestSet-2-MEDIQA-Sum-2023.csv", "test")]
    files += [(f[len(mts) + 1:], "train") for f in sorted(glob.glob(f"{mts}/Augmented-Data/*.csv"))]
    for f, split in files:
        for r in csv.DictReader(open(f"{mts}/{f}", encoding="utf-8-sig")):
            head = MTS_MAP.get(r["section_header"].strip().upper())
            if head:
                out[split].append({"id": f"mts-{f.split('/')[-1][:-4]}-{r['ID']}", "source": "mts",
                                   "dialogue": render(mts_turns(r["dialogue"])), "sections": [(head, r["section_text"].strip())]})
    if synth_glob:
        recs = [json.loads(l) for f in sorted(glob.glob(synth_glob)) for l in open(f)]
        random.Random(0).shuffle(recs)
        n_hold, dropped = max(1, len(recs) // 20), 0         # 5% held out: dev and test halves, with gold facts
        for i, r in enumerate(recs):
            split = "test" if i < n_hold // 2 else "dev" if i < n_hold else "train"
            # verifier filter on TRAIN only: dev/test stay an unbiased sample, so checker metrics are not pre-selected.
            # >= 2 flags: a single flag per note is about the checker's own false-alarm rate (~3%/clause x ~10 clauses)
            if split == "train" and sum(c.verdict == "contradicted" for c in check_note(
                    "\n".join(r["note"].values()), [(t["speaker"], t["text"]) for t in r["dialogue"]])) >= 2:
                dropped += 1
                continue
            out[split].append({"id": r["id"], "source": "synthetic", "facts": [positive_fact(f) for f in r["facts"]],
                               "dialogue": render((t["speaker"], t["text"]) for t in r["dialogue"]),
                               "sections": [(h, t.strip()) for h, t in r["note"].items()]})
        print(f"synthetic: {len(recs)} records, {dropped} train notes dropped by the verifier filter")
    aci_train = [r for r in out["train"] if r["source"].startswith("aci")]
    out["train"] += aci_train * (ACI_REPEAT - 1)          # 67 real full notes vs ~20k short synthetic ones
    test_dialogues = {r["dialogue"] for r in out["test"]}
    n = len(out["train"])
    out["train"] = [r for r in out["train"] if r["dialogue"] not in test_dialogues]   # 2 MTS back-translations collide
    print("dropped from train (dialogue also in test):", n - len(out["train"]))
    for split, rows in out.items():
        rows = [r for r in rows if r["dialogue"] and r["sections"]]
        assert not ({r["dialogue"] for r in rows} & {r["dialogue"] for r in out["train"]} if split == "test" else set()), \
            "test dialogue also in train"
        with open(f"{data}/{split}.jsonl", "w") as fo:
            fo.writelines(json.dumps(r) + "\n" for r in rows)
        print(split, len(rows), {s: sum(r["source"].startswith(s) for r in rows) for s in ("aci", "mts", "synthetic")})


if __name__ == "__main__":
    main(*sys.argv[1:])
