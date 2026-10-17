"""Score a candidate against GATE.md from two scripts.evaluate outputs (greedy block only).

usage: python -m scripts.gate_report BASELINE_results.txt CANDIDATE_results.txt
Pass = the candidate is strictly better on all three: ACI test1 unsupported (lower), ACI test1 ROUGE-1 (higher),
synthetic absent_asserted (lower).
"""
import re
import sys

METRICS = [("ACI-Bench test1: unsupported clauses %", "aci-test1", r"unsup=([\d.]+)%", min),
           ("ACI-Bench test1: ROUGE-1", "aci-test1", r"R1=([\d.]+)", max),
           ("Synthetic: denied facts asserted present %", "synthetic", r"absent_asserted=([\d.]+)%", min)]


def greedy_metrics(text):
    block = re.search(r"### \S*pred_greedy\.jsonl[^\n]*\n(.*?)(?=^### |\Z)", text, re.S | re.M).group(1)
    rows = {l.split()[0]: l for l in block.splitlines() if l.strip()}
    return [float(re.search(pat, rows[src]).group(1)) for _, src, pat, _ in METRICS]


def report(base_text, cand_text):
    base, cand = greedy_metrics(base_text), greedy_metrics(cand_text)
    wins = [b != c and best(b, c) == c for b, c, (*_, best) in zip(base, cand, METRICS)]
    lines = ["| Metric | Baseline | Candidate | Better |", "|---|---|---|---|"]
    lines += [f"| {name} | {b} | {c} | {'yes' if w else 'NO'} |" for (name, *_), b, c, w in zip(METRICS, base, cand, wins)]
    return all(wins), "\n".join(lines + ["", f"**Gate: {'PASS' if all(wins) else 'FAIL'}**"])


if __name__ == "__main__":
    print(report(open(sys.argv[1]).read(), open(sys.argv[2]).read())[1])
