# Gate: does the ~220M pretrained model replace the 97.5M one?

Fixed 2026-10-17, before the 220M model was pretrained, fine-tuned or evaluated.

## Candidate

- About 220M parameters: Llama-style decoder, 16 layers, d=1024.
- Tokenizer and weights both trained from scratch:
  - pretraining on FineWeb-Edu `sample/10BT` shards 000–003 (ODC-By), about 2.8B tokens;
  - then fine-tuning on the same `train.jsonl` as the 97.5M model.

## Baseline

`scratch90m`: 97.5M parameters, results in the README.

## Rule

The 220M model becomes the headline only if it beats the baseline on **all three** of the metrics below.
Both models use greedy decoding, the same test files, and the same `scripts/evaluate.py`.

| Metric | Baseline | Bar |
|---|---|---|
| ACI-Bench test1: unsupported clauses | 84.2% | lower |
| ACI-Bench test1: ROUGE-1 | 43.99 | higher |
| Synthetic test (200): denied facts asserted present | 7.6% (66/869) | lower |

If it misses any of the three, the README keeps the 97.5M model as the headline and reports the 220M model as a
negative result, with its numbers.

## Reporting rules

- Report CPU latency (4 threads, full note) for both models.
- Don't claim a head-to-head win over any closed model that publishes no benchmark numbers.

## Note added 2026-10-17, before any pretraining job ran

The candidate has 239.1M parameters, not ~220M: 16 layers, d=1024, 32k vocabulary. The rule above is unchanged.
