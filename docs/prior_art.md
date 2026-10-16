# Prior art: small dialogue-to-note models, grounding checkers, open data

Searched 2026-10-15 (web, GitHub API, Hugging Face API, figshare API). Licences were read from
LICENSE files, PyPI metadata or the API, unless a line says **unverified**. Where nothing turned
up, the text says "none found in these searches". That is not a claim that none exists.

## 1. Small (<1B) dialogue-to-note models and their numbers

**Nothing found is trained from scratch.** Every small public number comes from a pretrained
seq2seq checkpoint (BART, BioBART, LED) or a small instruct LLM (SmolLM2). groundnote therefore
has no like-for-like comparator. A closed commercial 175M from-scratch clinical model exists
(seen in a LinkedIn post), but it publishes no weights and no benchmark.

**ACI-Bench** ([repo](https://github.com/wyim/aci-bench), data CC BY 4.0 per README; the GitHub
API reports no licence file). 207 encounters: train 67 / valid 20 / test1 40 (= MEDIQA-Chat 2023
Task B test) / test2 40 (= Task C) / test3 40 (= MEDIQA-Sum 2023 Task C). Baselines come from
`tables/test1.csv`. The scale is 0-100, and the "Fact" column is **MEDCON**:

| test1, full note | R-1 | R-2 | R-L | BERTScore | MEDCON |
|---|---|---|---|---|---|
| copy whole transcript (no model) | 32.84 | 12.53 | 30.61 | 54.10 | **55.65** |
| BART-large (406M), full note | 41.76 | 19.20 | 34.70 | 63.89 | 43.38 |
| BART-large + FT-SAMSum, **Division** | **53.46** | **25.08** | **48.62** | 68.63 | 48.23 |
| BioBART-v2-large, Division | 49.53 | 22.47 | 44.92 | 67.38 | 43.06 |
| LED-large-16384, Division | 34.15 | 8.01 | 29.80 | 55.64 | 32.67 |
| GPT-4 (zero-shot) | 51.76 | 22.58 | 45.97 | 66.57 | **57.78** |

- Checkpoints confirmed from `baselines/*.sh`: `facebook/bart-large`, `GanjinZero/biobart-v2-large`,
  `allenai/led-large-16384` and `patrickvonplaten/led-large-16384-pubmed`. The parameter counts
  are the usual sizes for those checkpoints, not figures stated in the paper.
- **"Division" means generating section by section, and it wins by 1-14 points.** This matters
  most for a short-context small model: generate per section, not the whole note at once.
- Test2 and test3 show the same pattern. Best BART Division R-1 is 52.63 on test2 and 52.77 on
  test3.
- Copying the transcript scores MEDCON 55.65, near GPT-4's 57.78. **Concept recall can be gamed
  by verbosity.** A grounding or precision metric is the missing half.
- **Shared-task results are not on the same scale.** The MEDIQA-Sum 2023 overview
  ([MSR page](https://www.microsoft.com/en-us/research/?p=1098669)) reports a best R-1 F1 of
  **0.43** on subtask B (MTS-Dialog section notes) and **0.49** on subtask C (full note,
  ACI-Bench test3). Those use a 0-1 F1 and a different scorer, and teams were ranked on the mean
  of ROUGE-1, BERTScore and BLEURT. They are not comparable with 52.77 above.
  `tables/clinicalnlp_taskB_test1.csv` and `clef_taskC_test3.csv` are mostly NA and hold no
  team results.
- MEDIQA-Chat 2023 Task B: WangLab placed **2nd with a fine-tuned LED-large-16384-pubmed
  (~460M)** and 1st with GPT-4 in-context learning
  ([arXiv 2305.02220](https://arxiv.org/abs/2305.02220);
  [repo, MIT](https://github.com/bowang-lab/MEDIQA-Chat-2023-WangLab)). The checkpoint
  `wanglab/task-b-led-large-16384-pubmed-run-3` is on HF with no licence tag. GersteinLab placed
  4th with R-1 0.4011 ([arXiv 2305.05001](https://arxiv.org/abs/2305.05001)). The official Task B
  per-team table was **not retrieved**; it is in ACL Anthology 2023.clinicalnlp-1.52.
- MTS-Dialog ([repo](https://github.com/abachaa/MTS-Dialog), CC BY 4.0 verified): 1,201 train /
  100 valid / two tests of 200, plus 3.6k back-translated pairs. Each pair is one **section**,
  not a full note. Per-model numbers from the EACL 2023 paper were not retrieved.
- Small public checkpoints, none with a stated licence or eval:
  `dhananjay2912/lsg-bart-base-4096-mediqa-chat-taskb[-sectionwise-*]` (BART-base, about 140M,
  one model per section) and `fahmiaziz/SmolLM2-135M-Instruct-Clinical-Note`. The GGUF
  re-uploader's card says the latter "invents details that are not in the input" and does not
  disclose its training data. The Phi/Mistral/Gemma ACI-Bench or MTS-Dialog LoRAs on HF are
  all >=2B with no licence tags.
- Closest in spirit: [Ambient_Scribing_with_Synthetic_Data_Generator](https://github.com/Alirezaprogramerrd99/Ambient_Scribing_with_Synthetic_Data_Generator)
  (teacher-generated data, sub-4B SLM fine-tune, NHS framing). It has no licence and no
  published ACI-Bench numbers.
- Larger reference points: MediGen (Llama3-8B FT) reports an average R-1 of 58.22 on ACI-Bench
  (**unverified**, search snippet only). LLaMA-Clinic is 13B ([arXiv 2405.00715](https://arxiv.org/abs/2405.00715)).

**Bar to beat:** a ~100M from-scratch model is credible if it gets near BART-large's
48-53 R-1 (Division) at roughly a quarter of the parameters, **and** shows lower
unsupported-claim rates than the copy-transcript and BART baselines.

## 2. Faithfulness, hallucination and negation checkers

**Highest-value find:** [MTS-Dialog `Correlation-Study/`](https://github.com/abachaa/MTS-Dialog/tree/main/Correlation-Study)
(CC BY 4.0) has **manual fact-level hallucination and omission counts for 400 automatic
summaries** (4 models x 100 validation dialogues). It is a ready-made, human-labelled test set
for the grounding checker. Report checker precision and recall on it.

| Tool | Licence (verified) | Size / deps | Fit for groundnote |
|---|---|---|---|
| [medspaCy](https://github.com/medspacy/medspacy) ConText + Sectionizer | MIT (PyPI 1.3.1, active, pushed 2026-06) | spaCy 3.8, PyRuSH, pysbd, **medspacy_quickumls** (always installed; UMLS needed only if used) | **Reuse.** Negation, hypothetical, historical and family-history modifiers on note clauses and within an utterance. The Sectionizer can parse note headers. |
| [negspacy](https://github.com/jenojp/negspacy) | MIT (1.1.0) | spaCy only | Lighter; NegEx triggers only, no other ConText modifiers. |
| [pyConTextNLP](https://github.com/chapmanbe/pyConTextNLP) | Apache-2.0 per setup.py, no LICENSE file | idle since 2023 | Superseded by medspaCy. Skip. |
| [SummaC](https://github.com/tingofurro/summac) | Apache-2.0 | NLI backbones; PyPI 0.0.4 pins `huggingface-hub<=0.17.0` | Will conflict with modern transformers. Re-implement its zero-shot sentence-NLI scoring if wanted. |
| [AlignScore](https://github.com/yuh-zha/AlignScore) | MIT (repo + `yzha/AlignScore` weights) | RoBERTa-base 125M / large 355M | CPU-feasible base model. General-domain, untested on dialogue negation. |
| [HHEM-2.1-Open](https://huggingface.co/vectara/hallucination_evaluation_model) | Apache-2.0 | flan-t5-base, 110M | Lightest NLI-style judge; custom code (`trust_remote_code`). |
| [MiniCheck](https://github.com/Liyan06/MiniCheck) | repo Apache-2.0; RoBERTa-L / DeBERTa-L / Flan-T5-L weights MIT | 355M-783M | Sentence-level claim checking. Bespoke-MiniCheck-7B has no licence tag and is too big anyway. |
| [LettuceDetect](https://github.com/KRLabsOrg/LettuceDetect) | MIT | ModernBERT-base, 150M; the pip package pulls `openai` | **Span-level** unsupported-token tagging, the closest output format to "flag this clause". Load the HF model directly. Trained on RAG data, not dialogue. |
| MEDCON (ACI-Bench metric) | needs a **UMLS licence** (QuickUMLS + UMLS 2022AA, UTS account) | scispaCy, QuickUMLS | Cannot ship in an offline open checker. Use it for paper comparability only, and state that. |
| Clinical NLI on MedNLI | MedNLI is PhysioNet-credentialed (MIMIC-derived), from memory | n/a | **Avoid**: weights trained on it inherit the credentialed-data terms. |
| Med-HALT ([arXiv 2307.15343](https://arxiv.org/abs/2307.15343)) | none found (`medhalt/medhalt.github.io`) | n/a | Medical-exam QA hallucination benchmark, **not** a summary checker. Not relevant. |

**Reuse or hand-write?** Reuse medspaCy ConText for clause-level negation and certainty in the
note and in single dialogue turns. Hand-write the part no library above handles:
**cross-turn question-answer polarity**. In "Does your knee lock?" / "No, it doesn't", the
concept is in the doctor's turn and the negation is a bare answer in the patient's turn. NegEx
and ConText work inside one sentence, so they miss it. This is a small rule set (yes/no answer
detection plus antecedent-question concept binding), and it is the project's differentiator.
An NLI model (HHEM 110M or AlignScore-base 125M) is an optional second opinion behind a flag,
not the core.

## 3. Open dialogue-to-note data beyond ACI-Bench and MTS-Dialog

Cite and download from the **original source only**. HF licence tags on re-uploads are not
authoritative.

| Dataset | Licence | What it is | Verdict |
|---|---|---|---|
| [PriMock57](https://github.com/babylonhealth/primock57) | **CC BY 4.0** (LICENSE.md read) | 57 mock UK GP consults: audio, transcripts, doctor-written notes | **Use as OOD test set** (different note style, n=57), not for training |
| Fareez et al. 2022 OSCE ([figshare 16550013](https://doi.org/10.6084/m9.figshare.c.5545842.v1)) | **CC0** (figshare API) | ~272 simulated OSCE interviews, respiratory-heavy, transcripts and audio, **no notes** | **Use** as dialogue input for teacher-generated notes. History-taking is dense in yes/no negations, so it doubles as negation test material. |
| [SIMORD](https://huggingface.co/datasets/microsoft/SIMORD) (MEDIQA-OE 2025) | CDLA-Permissive-2.0 | Medical-order annotations on ACI-Bench + PriMock57 transcripts | Optional: gold labels for checking Plan-section orders |
| [MedSynth](https://huggingface.co/datasets/Ahmad0067/MedSynth) (10k+ synthetic pairs) | **none stated** on card | LLM-generated | Reject (unclear) |
| MediTOD ([dair-iitd](https://github.com/dair-iitd/MediTOD)) | no licence file | Annotations on the Fareez dialogues | Reject; use Fareez directly |
| [AGBonnet/augmented-clinical-notes](https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes) | tagged MIT | Built on NoteChat dialogues + PMC-Patients notes (CC BY-NC-SA, from memory) + GPT-4 output | Reject (inherits NoteChat / NC) |
| `nisten/opus5-5-doctor-patient-conversations-*` | tagged Apache-2.0 | Conversations generated by a proprietary model, no notes | Reject, same bucket as omi-health (generator terms unclear) |
| `har1/MTS_Dialogue-Clinical_Note` (MIT), `AxiomicLabs/ACI-Bench` (Apache), `ClinicianFOCUS/ACI-Bench-Refined` (AGPL) | relabelled re-uploads | Copies of CC BY 4.0 data | Do not use; cite the originals |
| `mkieffer/ACI-Bench`, `galgantar/MTS-Dialog-cleaned` | CC BY 4.0 | Mirrors | OK as convenience mirrors |
| CliniKnote ([arXiv 2408.14568](https://arxiv.org/abs/2408.14568)), MIMIC-derived sets, MTSamples | not found / credentialed / site terms | n/a | Reject |

**Teacher-model licence:** the teacher's licence governs whether its outputs may train
groundnote. Some open-weight licences add naming or derivative obligations; from memory, Llama
3.x requires a "Llama" name prefix and Gemma's terms count distillation outputs as derivatives.
**Verify per teacher.** Apache-2.0 teachers (e.g. most Qwen2.5 sizes, OLMo) are the clean
choice.

## 4. Where groundnote can stand out

Most open "AI scribe" repos found here wrap a hosted or local general LLM and add no
measurement:

- [OpenScribe](https://github.com/sammargolis/OpenScribe) (MIT, 217 stars): Whisper + Anthropic API key.
- [phlox](https://github.com/bloodworks-io/phlox) (MIT): local-first, any OpenAI-compatible endpoint, Ollama or llama.cpp.
- [AI-Scribe](https://github.com/1984Doc/AI-Scribe) (GPL-3.0): Whisper + KoboldCpp or GPT.

None of the repos sampled report ACI-Bench numbers or a hallucination rate.

Gaps groundnote can fill:

1. **From-scratch small model with published weights and a training recipe.** None found
   public; the only from-scratch precedent is closed.
2. **Dialogue-aware negation checking.** Cross-turn question-answer polarity, which NegEx,
   ConText, NLI and span detectors do not model. Ship a labelled negation-flip test suite.
3. **Checker validated against human labels.** Report precision and recall on the MTS-Dialog
   400-summary hallucination/omission annotations, not just ROUGE.
4. **Fully offline, UMLS-free evaluation.** MEDCON needs a UMLS licence; a grounding precision
   score that ships in the repo fills that hole and answers the copy-the-transcript
   loophole (MEDCON 55.65).
5. **Honest reporting.** Section-wise ("Division") ROUGE on ACI-Bench test1-3 against BART-large
   Division, an OOD score on PriMock57, CPU latency and RAM, and per-section unsupported-clause
   rates.
6. **Clean licence chain.** CC BY 4.0 + CC0 data, an Apache-2.0 teacher and MIT tools, all
   documented. Many HF mirrors and fine-tunes found here lack or mislabel licences.

Not checked: MTS-Dialog paper per-model ROUGE, the official MEDIQA-Chat Task B per-team table,
and the MediGen figure. Whether medspaCy installs cleanly on Windows or macOS CPU was not
tested.
