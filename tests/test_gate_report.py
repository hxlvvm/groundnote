from scripts.gate_report import report

BASE = """### /x/scratch90m/pred_greedy.jsonl  n=705  median sec/note=0.72
aci-test1  n= 40 R1=43.99 R2=17.73 RLsum=40.77 | pred: contra=2.7% unsup=84.2% gold: contra=3.8% unsup=37.0%
synthetic  n=200 R1=73.11 R2=51.10 RLsum=69.89 | pred: contra=2.1% unsup=45.4% gold: contra=2.3% unsup=23.3% | fact_recall=60.4% absent_asserted=7.6% (66/869)

### /x/scratch90m/pred_bo4.jsonl  n=240  median sec/note=1.16
aci-test1  n= 40 R1=99.00 R2=15.61 RLsum=33.91 | pred: contra=1.7% unsup=1.0% gold: contra=3.8% unsup=37.0%
synthetic  n=200 R1=72.97 R2=50.92 RLsum=69.77 | pred: contra=1.2% unsup=36.8% gold: contra=2.3% unsup=23.3% | fact_recall=62.7% absent_asserted=0.1% (56/869)
"""


def test_gate_uses_greedy_block_and_needs_all_three():
    better = BASE.replace("R1=43.99", "R1=45.00").replace("unsup=84.2%", "unsup=70.0%").replace("asserted=7.6%", "asserted=5.0%")
    assert report(BASE, better)[0]
    assert not report(BASE, better.replace("R1=45.00", "R1=43.99"))[0]      # a tie is not a win
    assert not report(BASE, BASE.replace("unsup=84.2%", "unsup=70.0%"))[0]  # bo4 block's numbers are ignored
