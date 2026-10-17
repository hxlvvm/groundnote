# groundnote

This is a small clinical-note model trained from scratch, together with a checker that reads every clause of the
note back against the transcript.

- **Input:** a doctor–patient conversation.
- **Output:** a sectioned note: Chief Complaint, HPI, Past Medical History, Allergies, Exam, Assessment, Plan.
- **Checker:** marks each clause *supported*, *unsupported* (not found in the transcript) or *contradicted* (found,
  but with the opposite polarity).

The checker exists for one failure in particular. A patient says "No, it doesn't" when asked "Has your knee locked?",
and a small model writes "the knee locked." The negation sits in the answer while the content words sit in the
question, so a bag-of-words match calls that clause supported. `groundnote/check.py` uses question+answer windows,
tracks polarity clause by clause, and lets the latest statement win. It is deterministic, needs no dependencies, and
runs offline next to the model.

> Research prototype. Not for clinical use. All transcripts are public benchmark data or synthetic.

## Results

**Model:** 239M-parameter Llama-style decoder.
- 16 layers, d=1024, 32k byte-level BPE vocabulary.
- Tokenizer and weights both trained from scratch, in two stages:
  - pretraining on 3.0B tokens of FineWeb-Edu, one epoch, 13.3 GPU-hours (mostly H100, in 30-minute chained jobs);
  - fine-tuning on the note data for 4 epochs, 24 minutes on one H100, keeping the best dev-loss epoch (3).
- Notes are generated section by section under forced headers. In ACI-Bench this "Division" setup beats
  whole-note generation by 1–14 ROUGE points ([prior art](docs/prior_art.md)).

It replaced an earlier 97.5M model, trained without pretraining, because it beat that model on all three metrics of
a bar fixed in advance in [GATE.md](GATE.md):

| Gate metric | 97.5M | 239M |
|---|---|---|
| ACI-Bench test1: unsupported clauses | 84.2% | **71.3%** |
| ACI-Bench test1: ROUGE-1 | 44.0 | **48.4** |
| Synthetic: denied facts asserted present | 7.6% (66/869) | **3.2%** (28/869) |

**Greedy decoding, held-out test splits that were never trained on:**

| Test set | n | ROUGE-1 | ROUGE-2 | ROUGE-Lsum | Unsupported clauses, model / gold note | Contradicted clauses, model / gold note |
|---|---|---|---|---|---|---|
| ACI-Bench test1 | 40 | 48.4 | 22.8 | 44.7 | 71% / 37% | 4.4% / 3.8% |
| ACI-Bench test1–3 | 120 | 48.7 | 22.8 | 45.2 | 72% / 37% | 4.4% / 5.0% |
| MTS-Dialog tests (one section each) | 385 | 41.4 | 23.3 | 37.1 | 89% / 54% | 1.3% / 3.0% |
| Synthetic (same teacher as train) | 200 | 88.6 | 74.7 | 86.1 | 25% / 23% | 2.2% / 2.3% |

**Published ACI-Bench test1 baselines, for scale.** Those papers use the official scorer; the ROUGE above comes from
`rouge-score` with stemming over the full note. The numbers are close, but not like-for-like.

| ACI-Bench test1 baseline | ROUGE-1 |
|---|---|
| copy the transcript | 32.8 |
| BART-large, whole note | 41.8 |
| BART-large + SAMSum, Division | 53.5 |
| GPT-4, zero-shot | 51.8 |

**How to read the table:**
- **The headline is the unsupported rate, not ROUGE.** On real conversations, 71% of the model's clauses cannot be
  found in the transcript. For gold notes the figure is 37%, because doctors paraphrase and add exam findings. That
  gap is the hallucination problem. On synthetic data the model is close to gold (25% vs 23%), so most of the gap is
  domain shift: 63% of training examples are synthetic (18,004 of 28,454).
- **Detected contradictions are 4.4% of clauses on ACI test1** (gold 3.8%, 97.5M model 2.7%). The rise is probably
  because more clauses now match the transcript closely enough to reach the polarity check. Most clauses still never
  do, and value swaps (golf instead of football) count as unsupported, not contradicted.
- **The synthetic rows are scored against gold fact labels, not the checker:**
  - fact recall 83.5%;
  - 3.2% of facts the patient *denied* appear in the note as present (28 of 869), which is the knee-locked error.
- **Best-of-4 reranking with the checker** brings that to 2.9% (25/869) and recall to 84.5% on the same 200
  encounters. It costs 1.2× the decoding time and 3.4 ROUGE-1 on ACI test1 (45.1 vs 48.4).

**CPU, laptop-class.** 4 cores, the 5 ACI test1 encounters, full note including the 6–7 forced sections:

| per note | 239M | 97.5M |
|---|---|---|
| median | 32 s | 32 s |
| range | 21–46 s | 6–64 s |

- Five notes per model, and the two models write notes of different lengths, so read this as "about half a minute",
  not as a speed ranking.
- ACI transcripts are long, about 1,500 words each.
- The K/V cache is carried across sections, so the transcript is read once rather than once per section.
  That made notes 2–12× faster, with output identical word for word.

**Reference: what someone else's pretraining buys.** SmolLM2-135M was pretrained by Hugging Face on about 2T tokens,
so it is *not* from scratch. It was fine-tuned on the same data and scored with the same script (greedy):

| | 97.5M, no pretraining | 239M, our pretraining (3B tokens) | SmolLM2-135M, pretrained by HF |
|---|---|---|---|
| ACI test1: unsupported clauses (gold 37%) | 84.2% | **71.3%** | 72.3% |
| ACI test1: ROUGE-1 | 44.0 | **48.4** | 43.1 |
| MTS-Dialog: ROUGE-1 | 35.9 | 41.4 | **45.8** |
| Synthetic: fact recall | 60.4% | **83.5%** | 82.5% |
| Synthetic: denied facts asserted present | 7.6% | 3.2% | **2.3%** |
| CPU, 4 cores, median per long note | 32 s | 32 s | 92 s |

- **Our own pretraining did what the SmolLM2 run suggested it would.** The negation-flip error fell from 7.6% to
  3.2% and fact recall rose from 60% to 84%.
- **SmolLM2 is still ahead on two rows:** denied facts asserted present (20 vs 28 of 869, a small difference at this
  sample size) and MTS-Dialog ROUGE-1. It saw about 660× more pretraining text (2T vs 3.0B tokens).
- **SmolLM2 is about 3× slower on CPU:** 30 layers and a 49k vocabulary.

## Data

| Source | Use | Licence |
|---|---|---|
| [ACI-Bench](https://github.com/wyim/aci-bench) | 67 train (oversampled 20×) / 20 valid full notes; 3 × 40 test | CC BY 4.0 |
| [MTS-Dialog](https://github.com/abachaa/MTS-Dialog) | 1,201 train + back-translated augmentation; 2 test sets | CC BY 4.0 |
| Synthetic encounters | ~20k dialogue + note + fact list, 5% held out as dev/test | generated by [Gemma 4 31B-it](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct) (Apache-2.0) via vLLM |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | pretraining only: `sample/10BT` shards 000–003, 3.0B tokens; ~300 MB of it, plus the train set, trained the 32k tokenizer | ODC-By 1.0 |

**How the synthetic data was made:**
- Each encounter is one JSON object: a dialogue, a note, and a list of facts marked present or absent. Absent facts
  are what make the negation test possible.
- On train only, notes with two or more contradicted clauses were dropped. Dev and test stay unfiltered, so the
  checker's rates on them are not pre-selected.
- `groundnote/df.json` holds word document frequencies over the ACI-Bench and MTS-Dialog train dialogues.

**Picking the teacher.** [docs/teacher_options.md](docs/teacher_options.md) covers licences: hosted APIs whose
terms forbid training on their outputs were excluded.
- Gemma 4 31B, gpt-oss-120b, Qwen3 and GLM models were piloted on the same prompts.
- A late challenger had to pass a gate fixed before its pilot ran, from Gemma's own scores (`scripts/gate_teacher.py`):
  - contradicted ≤ 3%;
  - unsupported ≤ 30%;
  - ≥ 75% of numbers in the note actually spoken.
- GLM-4.5-Air failed it, at 58.7% unsupported and 56% grounded numbers, so its shards were never generated.

## Try it

- **Live demo** (being published; the link resolves once it is up):
  [huggingface.co/spaces/Opflare1144/groundnote](https://huggingface.co/spaces/Opflare1144/groundnote).
  A sample transcript is already filled in, and the free CPU takes up to a couple of minutes per note.
  **Do not paste real patient information.** On the hosted demo your text is sent to a Hugging Face server; use the
  sample or an invented visit.
- **Example output:** [a held-out real knee visit](https://htmlpreview.github.io/?https://github.com/hxlvvm/groundnote/blob/master/docs/example_knee.html)
  (ACI-Bench test1 D2N126, CC BY 4.0) and
  [an in-domain synthetic visit](https://htmlpreview.github.io/?https://github.com/hxlvvm/groundnote/blob/master/docs/example_synthetic.html).
  [The knee-locking visit](https://htmlpreview.github.io/?https://github.com/hxlvvm/groundnote/blob/master/docs/example_locking.html)
  is the fictional case the checker was built around. The note sits beside the transcript, every clause is coloured
  by its verdict, and hovering a clause shows the transcript turn it rests on. All three come from the 239M model.
- **Weights** (being published): [Opflare1144/groundnote-239m](https://huggingface.co/Opflare1144/groundnote-239m).

**Run it on your own machine.** CPU is enough, and when you run it locally, nothing leaves the machine (the hosted
demo sends your text to Hugging Face):

```bash
pip install torch==2.13.0 transformers==5.17.0
python -m groundnote.app Opflare1144/groundnote-239m                   # web page at http://localhost:8000
python -m groundnote.scribe Opflare1144/groundnote-239m visit.txt --html note.html   # the same from the command line
```

A transcript is one turn per line, written as `Doctor: ...` / `Patient: ...`. A long visit takes about 30 s on a
laptop CPU. `space/` holds the Dockerfile for the Space.

## Train and evaluate

```bash
pip install -r requirements.txt                      # vllm is only needed to generate data

python -m scripts.prep_data DATA_DIR "DATA_DIR/full_*.jsonl"   # ACI-Bench + MTS-Dialog + synthetic -> train/dev/test.jsonl
python -m scripts.pretrain tok   PT_DIR DATA_DIR      # 32k BPE; FineWeb-Edu shards in DATA_DIR/fineweb (jobs/dl_corpus.sbatch)
python -m scripts.pretrain pack  PT_DIR DATA_DIR      # -> PT_DIR/tokens.bin + val.bin
python -m scripts.pretrain train PT_DIR runs/pt STOP_AT    # STOP_AT = epoch-seconds deadline; rerun resumes
python -m scripts.train runs/ft DATA_DIR --init runs/pt/final --lr 3e-4 --epochs 4 --batch 4 --grad-accum 4
python -m scripts.generate runs/ft/final DATA_DIR/test.jsonl pred.jsonl
python -m scripts.evaluate pred.jsonl
```

The 97.5M baseline skips pretraining: `python -m scripts.train runs/scratch DATA_DIR --epochs 8`. `jobs/` holds the
SLURM scripts that chain these steps in 30-minute links (`--stop-at` checkpoints and stops; a rerun resumes).

The synthetic corpus is not released yet. `python -m pytest` runs the tests.

## Limits

- The checker is lexical. It flags some inventions; green means lexical overlap only. On the real knee visit it
  passed pain on the medial side when the patient said the outside, and a plan of physical therapy nobody mentioned.
  - It marked "denies hearing a pop" as not found, not contradicted, although the patient said the knee popped.
  - It flagged a correct "denies numbness or tingling in her toes" as a contradiction: a later answer about the toes
    ("can you feel your toes?" "I think so") won the latest-statement rule.
  - It misses paraphrases with no shared content words ("BID" for "twice a day"), which inflates "unsupported" on
    every note, gold notes included. It catches some of the negation flips it was built for.
- The training data is mostly one teacher's style, which matches the gap between the synthetic and real-data rows.
- 120 ACI test encounters is a small sample; differences of 1–2 ROUGE points are within noise.

## Licence

MIT for the code. We release the weights under CC BY 4.0. Pretraining data: FineWeb-Edu (Hugging Face, ODC-By 1.0).
Fine-tuning data: ACI-Bench (Yim et al., 2023) and MTS-Dialog (Ben Abacha et al., 2023), both CC BY 4.0; we
oversampled ACI-Bench and used MTS-Dialog's back-translated augmentation. Synthetic conversations by Gemma 4 31B
(Apache-2.0 per its model card). The data sources keep their own
licences (above).
