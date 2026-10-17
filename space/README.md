---
title: groundnote
emoji: 🩺
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
license: mit
models:
  - Opflare1144/groundnote-97m
short_description: From-scratch note model with clause-level grounding check
---

# groundnote: live demo

Paste a doctor–patient transcript and get a sectioned clinical note. Every clause is underlined:

- **green:** supported by the transcript
- **amber:** not found in the transcript
- **red:** contradicts the transcript

The model has 97.5M parameters, all trained from scratch. On real conversations it still invents details, and the
checker is there to show them.

Code: https://github.com/hxlvvm/groundnote. Research prototype, not for clinical use. The free CPU Space takes up to
about a minute per note.
