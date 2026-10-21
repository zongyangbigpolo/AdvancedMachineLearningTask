# Final report and live demo

This is an appendix to the [Stage 5 README](README.md), not a separate stage or weekly task.

The eventual deliverable is an **English report in the required NeurIPS format**, plus a reproducible live demo. The [Stage 1 teaching report](../01-basics/reports/README.md) records an actual bounded 98M run, not full pretraining. Confirm the required year/template before adding the official style files.

The report should answer four concrete questions:

| Section | Required evidence |
| --- | --- |
| Problem | A precise task, data split, evaluation protocol and measurable acceptance criteria |
| Motivation | Why the question matters and what the baseline cannot establish |
| Method | Architecture, tokenizer, data provenance, objectives, compute budget and controlled comparisons |
| Results | Actual measurements, learning curves, uncertainty, ablations, failure cases and limitations |

Keep checkpoint identifiers, dataset fingerprints, commands, seeds and environment information with each result. Disclose AI assistance and external references according to the applicable course policy. Use the official NeurIPS template when preparing the final paper; this directory is not itself a formatted submission.

The first-stage demo should show tokenizer round-trip, causal behavior, a saved training curve, checkpoint resume and generation. A toy-model demo must be labeled as such. Later demos should separately establish kernel correctness/speed, distributed gains, scaling predictions, alignment comparisons and externally verified agent success.
