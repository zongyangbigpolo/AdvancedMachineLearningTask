# Final deliverables

This appendix belongs to the [stage-five experiment record](README.md).

| Deliverable | Saved artifact |
| --- | --- |
| English report, problem / motivation / method / results | [paper.pdf](report/paper.pdf), [paper.tex](report/paper.tex) |
| Official NeurIPS 2026 style and provenance | [neurips_2026.sty](report/neurips_2026.sty), [template source](report/TEMPLATE_SOURCE.txt) |
| Reloaded neural-policy demo | [demo.py](demo.py), [execution log](reports/demo.jsonl) |
| Replayable environment interactions | [traces.json](reports/traces.json) |
| Training data and saved agent weights | [agent-artifacts.zip](artifacts/agent-artifacts.zip) |

The compiled five-page report uses the official preprint style. It covers
including weak story generation, the constrained numeric response policy,
the finite-state agent environment, failed baselines, and absent
Triton/multi-GPU measurements.

The demo reloads `agent.pt`, executes eight sequential goals, prints each
tool transition, and exits with an error if the external verifier rejects
the final state. The saved fresh run completed 36 tool actions and 72
model calls with self-review enabled.

GitHub Copilot assisted implementation, execution, and drafting.
Method sources and experimental scope are documented in the report and
the [reference log](../01-basics/docs/references.md).
