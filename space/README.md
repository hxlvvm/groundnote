---
title: groundnote
emoji: 🩺
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
license: mit
models:
  - Opflare1144/groundnote-239m
short_description: From-scratch note model with clause-level grounding check
---

# groundnote: live demo

Paste a doctor–patient transcript and get a sectioned clinical note. Every clause is underlined:

- **green:** words found in transcript
- **amber:** not found in transcript
- **red:** contradicts transcript
- **grey:** unclear

**Do not paste real patient information.** On the hosted demo your text is sent to a Hugging Face server; use the
sample or an invented visit.

The model has 239M parameters, all trained from scratch: pretrained on 3.0B tokens of FineWeb-Edu, then fine-tuned
on clinical conversations. On real conversations it still invents details, and the checker flags some of them. The
checker is lexical: green means lexical overlap only, not that the clause is correct. On a real benchmark knee visit
it passed pain on the wrong side of the knee and a plan of physical therapy nobody mentioned.

Code: https://github.com/hxlvvm/groundnote. Research prototype, not for clinical use. The free CPU Space takes up to
a couple of minutes per note.
