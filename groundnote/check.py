"""Grounding check: is each clause of a generated note supported by the dialogue, and with the same polarity?

The failure this targets: the patient says "No, it doesn't" to "Has your knee locked?", and the note says
"the knee locked". The negation sits in the short answer while the content words sit in the question, so
evidence is a question+answer window and polarity is tracked clause by clause through the whole dialogue,
latest statement wins (that also handles "No. Actually, it did lock once last week.").

Deterministic and dependency-free: it runs offline on the same device as the model.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

STOP = set("""a an the and or of to in on at for with by from as is are was were be been being am do does did done
has have had having it its it's this that these those there here he she they them his her their him you your i me my we
our us mr mrs ms dr patient pt patients reports reported report states stated notes noted presents presented describes
described mentions mentioned says said complains complaint endorses endorsed also some any ever very just so then now
well okay ok um uh like really can could would should will may might about which who what when where how if
year years old today currently since due because into than too out up down over still yet get got""".split())
NEG = set("""no not never none nothing without denies denied deny denying neither nor cannot negative absent nobody
nowhere""".split())
NEG_ANYWHERE = {"negative", "absent"}           # can follow the concept: "fracture is absent"
DOUBLE_NEG = re.compile(r"\b(not|never|\w+n't)\s+(\w+\s+){0,2}(say|deny|denies|mean|think|claim)\b")
UNSURE = re.compile(r"\b(don't know|not sure|don't remember|can't remember|unsure|maybe|i guess|no idea)\b")
YES = re.compile(r"^(yes|yeah|yep|yup|uh-huh|mm-hmm|correct|right|sometimes|definitely|absolutely|it does|i do|it did|i did|i have)\b")
CLAUSE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\s*\b(?:but|actually|though|although|however|except)\b\s*")
SUPPORT_THRESHOLD = 0.6
NEG_SCOPE = 12                                  # NegEx-style: a cue negates up to 12 tokens on, unless a terminator intervenes
SCOPE_END = set("but however although though except so okay ok now then which who um uh alright yeah".split())
ASKING = re.compile(r"\b(any|ever|(do|does|did|are|is|have|has|were|was)\s+(you|your|it|he|she|they|there))\b")
PLAN = re.compile(r"\b(advis|recommend|will|plan|prescrib|order|schedul|refer|follow[- ]?up|counsel)", re.I)
QUESTION_START = re.compile(r"^(do|does|did|are|is|was|were|have|has|any|can|could|would|will|how|what|when|where|why|which)\b")
WINDOW = 3                                      # turns per evidence unit: topic, question, answer
DF = json.loads((Path(__file__).with_name("df.json")).read_text())["df"]   # built by scripts/build_df.py from train dialogues


def tokens(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", re.sub(r"\s+n't\b", "n't", text.lower()))   # ACI writes "do n't"


def is_neg(tok):
    return tok in NEG or tok.endswith("n't")


def stem(w):
    for suf in ("ies", "ing", "ed", "es", "s", "er", "ly"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break
    if w[-1] in "ey" and len(w) > 3:                 # injury/injured, allergy/allergies
        w = w[:-1]
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiou":
        w = w[:-1]
    return w


def content_stems(text):
    return {stem(t) for t in tokens(text) if t not in STOP and not is_neg(t)}


def parse_dialogue(text):
    """'[doctor] hi' / 'Doctor: hi' lines -> [(speaker, utterance)]. Unlabelled lines continue the previous turn."""
    turns = []
    for line in text.splitlines():
        m = re.match(r"\s*\[?([A-Za-z_]+)\]?\s*[:\]]?\s+(.*)", line)
        if m and m.group(1).lower() in {"doctor", "patient", "guest_family", "guest_clinician", "nurse", "family"}:
            turns.append((m.group(1).lower(), m.group(2).strip()))
        elif line.strip() and turns:
            turns[-1] = (turns[-1][0], turns[-1][1] + " " + line.strip())
    return turns


def split_clauses(text):
    return [c.strip() for c in CLAUSE_SPLIT.split(text) if c and c.strip(" .,;!?")]


def clause_polarity(clause, key):
    """+1 asserted, -1 negated, 0 asked about (unpunctuated speech), None if the clause does not mention the concept."""
    toks = tokens(clause)
    hits = [i for i, t in enumerate(toks) if stem(t) in key]
    if not hits:
        return None
    scope = toks[max(0, hits[0] - NEG_SCOPE): hits[0]]
    ends = [i for i, t in enumerate(scope) if t in SCOPE_END]
    scope = scope[ends[-1] + 1:] if ends else scope
    neg = any(map(is_neg, scope)) or any(t in NEG_ANYWHERE for t in toks[hits[0]: hits[0] + 6])
    if not neg and ASKING.search(" ".join(scope)):
        return 0
    # shortcut: one explicit double-negation pattern ("wouldn't say it never"), not a parser; upgrade if audits show misses
    if neg and DOUBLE_NEG.search(clause.lower()) and sum(map(is_neg, scope)) >= 2:
        neg = False
    return -1 if neg else 1


def bare_answer(clause):
    """Short yes/no answer with no content of its own: 'No, it doesn't.' / 'Not anymore.' / 'Yeah.'"""
    low = clause.lower().strip(" .,!")
    toks = tokens(low)
    if not toks or len(toks) > 6 or len(content_stems(low)) > 1 or UNSURE.search(low):
        return None
    if YES.match(low):
        return 1 if len(toks) <= 3 else None
    return -1 if any(is_neg(t) for t in toks) else None


def dialogue_polarity(turns, key):
    """Latest statement about the concept wins. Questions set context but assert nothing.
    Returns (polarity, index of the deciding turn)."""
    polarity, where, context = None, None, False
    for t, (_, utt) in enumerate(turns):
        for c in split_clauses(utt):
            if UNSURE.search(c.lower()) and context:
                polarity, where = None, t
                continue
            p = clause_polarity(c, key)
            if p is not None:
                context = True
                if p and not (c.rstrip().endswith("?") or QUESTION_START.match(c.lower())):
                    polarity, where = p, t
                continue
            b = bare_answer(c)
            if b is not None and context:
                polarity, where = b, t
            elif b is None:
                context = False
    return polarity, where


@dataclass
class Claim:
    text: str
    verdict: str            # supported | contradicted | unsupported | unclear
    evidence: tuple         # indices of the dialogue turns used as evidence
    coverage: float


def check_note(note, turns):
    """Split a note into clauses and grade each against the dialogue turns."""
    turn_stems = [content_stems(u) for _, u in turns]
    out = []
    for line in note.splitlines():
        if line.strip().isupper():                                   # bare section header
            continue
        line = re.sub(r"^\s*([-*•]|\d+\.)?\s*([A-Z][A-Za-z /&]+:)?\s*", "", line)   # bullets, "1.", "Chief Complaint:"
        for clause in split_clauses(line):
            s = {w for w in content_stems(clause) if not w.isdigit()}
            if not s:
                continue
            best, ev, near = 0.0, (), set()
            for i in range(len(turns)):
                idx = tuple(range(max(0, i - WINDOW + 1), i + 1))
                cov = len(s & set().union(*(turn_stems[j] for j in idx))) / len(s)
                if cov >= SUPPORT_THRESHOLD:
                    near.update(idx + (i + 1,) * (i + 1 < len(turns)))   # window + the answer that follows it
                if cov > best:
                    best, ev = cov, idx
            if best < SUPPORT_THRESHOLD:
                out.append(Claim(clause, "unsupported", ev, best))
                continue
            matched = s & set().union(*(turn_stems[i] for i in ev))
            rarest = min(DF.get(w, 0) for w in matched)
            key = {w for w in matched if DF.get(w, 0) == rarest}   # most specific matched term: "lock", not "knee"
            if PLAN.search(clause):
                # shortcut: orders/plans are not checked for polarity (patient habits read as false flips); revisit with an NLI judge
                out.append(Claim(clause, "supported", ev, best))
                continue
            note_p = clause_polarity(clause, key)
            order = sorted(near)
            dia_p, where = dialogue_polarity([turns[i] for i in order], key)
            where = None if where is None else order[where]
            if dia_p is None or note_p is None:
                verdict = "unclear"
            else:
                verdict = "supported" if dia_p == note_p else "contradicted"
            out.append(Claim(clause, verdict, (where,) if where is not None else ev, best))
    return out
