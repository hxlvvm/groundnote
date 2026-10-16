# Teacher LLM options for synthetic dialogue + note generation

Researched 2026-10-15. Job: 20k-50k cases × ~1,500 output tokens (30-75M tokens). Budget $0, H100 80GB,
1-4 GPUs per single-node job. The student model and the data get published openly.
Licence and size facts come from the Hugging Face API (`cardData.license`, `safetensors.total`) and the
model cards, all read 2026-10-15. **[unverified]** = not checked against a primary source.
**est.** = not measured on our cluster.

## Filter 1: may the outputs train a published model?

| Source | Licence / terms (quoted) | Outputs usable? |
|---|---|---|
| GLM-5, 5.1, 5.2, 4.5-Air, 4.6, 4.7, 4.7-Flash, 5.3-Flash weights | card `license: mit` | yes |
| GLM-5.3 weights | custom "GLM-5.3 License": MIT-style grant, plus a Z.ai security review only for "Model as a Service" operators with revenue above $10B | yes, for us |
| Qwen3-8B, 30B-A3B-Instruct-2507, 32B, 3.6-35B-A3B, 3.8-27B | `license: apache-2.0` | yes |
| gpt-oss-20b / 120b | `apache-2.0`, plus USAGE_POLICY: "you agree to comply with all applicable law" | yes |
| Gemma 4 31B / 26B-A4B | card says `apache-2.0`. Google's separate "Prohibited use" page **[not read]** | probably |
| Mistral Small 3.2 24B, Mistral Small 4 119B | `apache-2.0` | yes |
| **Z.ai hosted API** (free tier: GLM-4.7-Flash, 4.5-Flash, 4.6V-Flash at $0) | Additional Terms §1(xii): "any use of the Z.ai's models, prompts, or model-generated content for the development, training, labeling, fine-tuning, optimization, iteration, or similar activities related to external models is strictly prohibited." (last updated 2026-04-14) | **NO** |
| **Gemini API free tier** | "You may not use the Services to develop models that compete with the Services (e.g., Gemini API or Google AI Studio)." (effective 2026-03-23) | ambiguous, but moot (see below) |
| Excluded: Qwen3.8-Flash-Next (`qwen-community-1.0`), Mistral Medium 3.5 (`other`), MedGemma-27B (HAI-DEF terms, gated), Nemotron 3.5 Lightning (`openmdw-1.1`, **[not read]**) | not Apache/MIT | skip |

The Z.ai prohibition is in the API terms only. Self-hosted MIT GLM weights do not carry it.

### Gemini free tier: ruled out on quota, so the "compete" question never has to be settled
- Free models as of the pricing page (2026-10-14): Gemini 3.8 Flash, 3.6 Flash, 3.5/3.1 Flash-Lite,
  3 Flash Preview, 2.5 Pro/Flash/Flash-Lite, Gemma 4.
- Requests per day (RPD): the official rate-limit page (2026-10-13) no longer publishes free-tier figures.
  It only says "View your active rate limits in AI Studio". Limits apply per project.
- Third-party reports **[unverified]**: 250 RPD for 2.5 Flash in 2025, about 20 RPD after a
  December 2025 cut, about 100 RPD in April 2026. At 20-250 RPD, 20k requests take 80-1,000 days.
  Not feasible.
- Data use: free-tier prompts and responses are used to "improve, and develop" Google products, and
  human reviewers may read them. That does not apply to users in the EEA, Switzerland or the UK, where
  the Paid Services data terms cover free use as well.
- The terms also say "You may not use the Services in clinical practice, to provide medical advice".
  Generating synthetic training data is neither, but the clause is worth knowing about.

## Filter 2: fits one node, ideally 1-4 H100s
- **GLM-5 / 5.1 / 5.2 / 5.3** (744B total, 40B active per token): FP8 weights are about 754 GB, more than
  the 640 GB on 8×H100, so they do not fit one node in FP8. 4-bit `nvidia/GLM-5.x-NVFP4` repos exist,
  but NVFP4 is Blackwell-native and running it on Hopper is **[unverified]**. Either route needs ≥8 GPUs.
- **GLM-5.3-Flash** (320B total, 18B active, multimodal): FP8 about 328 GB, more than the 320 GB on
  4×H100. Needs 8×H100 (TP8). **GLM-4.6 / 4.7** (355B, 32B active): FP8 about 358 GB, also 8 GPUs.
- **No GLM-5-Air exists.** The GLMs that fit 1-2 GPUs are GLM-4.7-Flash (30B, 3B active, 1 GPU) and
  GLM-4.5-Air (106B, 12B active, FP8 about 110 GB, 2 GPUs).
- The installed `.venv` vLLM 0.31.0 registers every architecture below (grepped the registry):
  GlmMoeDsa, Glm5Next, Glm4MoeLite, Qwen3_5, Qwen3Moe, GptOss, Gemma4, Mistral3.

## Filter 3: cost and medical evidence for the survivors
GPU-hours per 30M *output* tokens, scaled up for reasoning tokens where the model emits them. All
throughput figures are **est.**: no published vLLM H100 numbers were found. Multiply by 2.5 for 75M.
The KV figure is the GPU memory left for the KV cache after weights; it caps how many sequences run
at once.

| Model (mode) | Weights / GPUs | KV headroom | est. GPU-h | Medical evidence |
|---|---|---|---|---|
| **gpt-oss-120b** (117B, 5.1B active; reasoning effort low, assumed +30% tokens) | MXFP4 ~61 GB, 1 | ~10 GB at ~37 KB/token (18 full-attention layers, 18 with a 128-token window): ~270k tokens, ~100 concurrent sequences (computed) | 3-6 | **HealthBench 53.0 low / 57.6 high; Hard 22.8 / 30.0** (arXiv 2508.10925 Table 3, verified) |
| gpt-oss-20b (21B, 3.6B active) | ~13 GB, 1 | large | 1-2 | HealthBench 40.4 low / 42.5 high (same table) |
| **Qwen3.8-27B-FP8** (Aug 2026, thinking off) | ~29 GB, 1 | ~45 GB; 48 of 64 layers use linear attention, ~65 KB/token | 3-6 | none on card |
| Qwen3.6-35B-A3B-FP8 (thinking off) | ~36 GB, 1 | ~40 GB | 1-2 | none on card |
| Qwen3-30B-A3B-Instruct-2507-FP8 (non-thinking model) | ~31 GB, 1 | ~45 GB | 1-2 | none on card |
| Qwen3-32B-FP8 (thinking off) | ~33 GB, 1 | ~40 GB | 3-6 | none on card |
| Qwen3-8B | ~16 GB, 1 | large | ~1 | none; weakest writer here |
| GLM-4.7-Flash (30B, 3B active; thinks by default) | BF16 ~62 GB, 1 (no official FP8) | ~10 GB, so use TP2 or online FP8 | 2-4 | none (τ²-Bench 79.5) |
| GLM-4.5-Air-FP8 (106B, 12B active) | ~110 GB, 2 | ~30 GB | 4-8 | none |
| Gemma 4 31B / 26B-A4B | BF16 62 / 52 GB, 1 | 10-20 GB | 3-6 / 1-2 | MedXPertQA **MM** 61.3 / 58.1 (multimodal QA, not dialogue) |
| Mistral Small 3.2 24B / Small 4 119B (6.5B active) | 48 GB, 1 / FP8 119 GB, 2 | ok | 3-5 / 3-6 | none |

Throughput does not decide between these: every survivor costs about 20 GPU-hours or less even at
75M tokens, so 4 one-GPU array shards finish in a few hours. Quality decides. HealthBench scores
rubric-graded health *conversations*, so it is the only benchmark here that resembles the task.
Only gpt-oss publishes it.

## Recommendation: gpt-oss-120b, reasoning effort low, 1 H100 per shard

Why:
1. Apache-2.0 with no restriction on using outputs.
2. The only candidate with verified evidence on health conversations: HealthBench 53.0 even at low
   effort. The model card says it nearly matches o3 at high effort.
3. One GPU per shard, ~100 concurrent sequences. Already supported by the installed vLLM.

Weaknesses you must manage:
- **Reasoning overhead.** The +30% token assumption is unmeasured.
- **Higher hallucination.** PersonQA hallucination rate is 0.491 vs o4-mini's 0.361 (Table 9).
  Check that drugs and doses in each note are consistent with its dialogue, rather than trusting
  free recall.
- **Structured output.** vLLM guided JSON combined with the gpt-oss reasoning parser is
  **[unverified]** on 0.31. Fallback: free-form JSON, then pydantic validation and a retry.

**Challenger: Qwen3.8-27B-FP8 with `enable_thinking: False`.** It is newer (Aug 2026), has more KV
headroom and no reasoning tax, but no medical evidence. Check whether the `--language-model-only`
flag shown on the Qwen3.6 card also applies to 3.8.

**Next step: a pilot, not part of this task.** One sbatch per model, 1 GPU, ~30 min, 200 identical
case seeds. Measure:
- JSON-valid rate;
- decoded tokens/s, which replaces every est. above;
- a blind spot-check of 30 cases for fact and polarity agreement between dialogue and note.

Adopt gpt-oss-120b unless Qwen3.8 clearly wins on JSON validity or the spot-check. Splitting the
corpus across both teachers is a cheap way to add style diversity if they tie. Name the teacher
model and its licence in the dataset card.
