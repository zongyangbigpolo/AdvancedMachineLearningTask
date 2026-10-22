"""Inspect a checkpoint and render training logs for the experiment report."""

import argparse
import html
import json
import shutil
from pathlib import Path

import torch

from aml.data import sha256


COMMANDS = """mkdir -p runs # Working directory: stages/01-basics
.venv/bin/python -m aml.tinystories --output data/week1-raw --train-rows 512 --val-rows 128 > runs/week1-data-download.log
.venv/bin/python -m aml.tokenizer --input data/week1-raw/train.txt --output data/week1-tokenizer.json --vocab-size 8192
.venv/bin/python -m aml.data --train data/week1-raw/train.txt --val data/week1-raw/val.txt --tokenizer data/week1-tokenizer.json --output data/week1-tokens > runs/week1-data-encode.log
.venv/bin/python -m aml.train --config configs/100m-mac.json --data data/week1-tokens --output runs/week1-100m-mac --device mps --stop-after 5 > runs/week1-training-first.log
.venv/bin/python -m aml.train --config configs/100m-mac.json --data data/week1-tokens --output runs/week1-100m-mac --device mps --resume > runs/week1-training-resume.log
.venv/bin/python -m aml.generate --checkpoint runs/week1-100m-mac/latest.pt --prompt 'Once upon a time' --temperature 0 --max-new-tokens 64 --device mps > runs/week1-generation.txt

# Checkpoint inspection and report rendering
.venv/bin/python -m scripts.evidence > reports/checkpoint-verification.log
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/week1-100m-mac"))
    parser.add_argument("--report", type=Path, default=Path("reports"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    summary = json.loads((args.report / "summary.json").read_text())
    manifest = json.loads((args.run / "manifest.json").read_text())
    checkpoint_path = args.run / "latest.pt"
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if state["config"] != manifest["config"] or state["step"] != summary["steps"]:
        raise ValueError("Checkpoint/config/summary disagreement")
    checkpoint_sha = sha256(checkpoint_path)
    if checkpoint_sha != summary["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint does not match the recorded hash")
    parameters = sum(value.numel() for value in state["model"].values())
    if parameters != summary["parameters"]:
        raise ValueError("Checkpoint parameter count disagrees with report")
    if not all(torch.isfinite(value).all().item() for value in state["model"].values()):
        raise ValueError("Non-finite checkpoint weights")
    optimizer_steps = sorted({int(value["step"].item()) for value in state["optimizer"]["state"].values()})
    if optimizer_steps != [summary["steps"]]:
        raise ValueError("Adam states do not match the completed update budget")
    for name, digest in manifest["source_sha256"].items():
        if sha256(Path("aml") / name) != digest:
            raise ValueError(f"Training source has changed: {name}")
    output = args.report / "evidence"
    output.mkdir(parents=True, exist_ok=True)
    first = Path("runs/week1-training-first.log")
    resumed = Path("runs/week1-training-resume.log")
    rows = [json.loads(line) for path in (first, resumed) for line in path.read_text().splitlines()]
    if rows != [json.loads(line) for line in (args.run / "metrics.jsonl").read_text().splitlines()]:
        raise ValueError("Original stdout logs disagree with the training metrics")
    if [row["step"] for row in rows] != list(range(summary["steps"] + 1)):
        raise ValueError("Missing or duplicate training steps")
    for path in (first, resumed):
        shutil.copyfile(path, output / path.name)
    audit = {
        "verification": "Checkpoint inspection passed",
        "checkpoint": str(checkpoint_path), "sha256": checkpoint_sha,
        "bytes": checkpoint_path.stat().st_size, "actual_weight_parameters": parameters,
        "completed_step": state["step"], "adam_parameter_states": len(state["optimizer"]["state"]),
        "adam_steps": optimizer_steps, "all_weights_finite": True,
        "source_fingerprints_match": True, "original_stdout_matches_metrics": True,
        "recorded_device": state["device"], "verification_device": "cpu",
        "report_type": "Checkpoint and optimizer inspection",
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (output / "commands.txt").write_text(COMMANDS)
    selected = "\n".join(json.dumps(row, indent=2) for row in rows if row["step"] in (0, 5, 6, 100))
    generation = (args.report / "generation.txt").read_text()
    document = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Stage 1: 98M training report</title>
<style>
body{{font:16px/1.5 system-ui,sans-serif;margin:0;background:#edf1f5;color:#172033}}
main{{max-width:1120px;margin:24px auto}} section{{background:white;padding:28px;margin:20px 0;border-radius:12px}}
h1,h2{{margin-top:0}} pre{{background:#122033;color:#e3edf9;padding:18px;white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 monospace}}
.notice{{border-left:5px solid #d97706;padding:12px;background:#fff7e6}} img{{max-width:100%}} a{{color:#174ca6}}
</style><main>
<section id="results"><h1>98M Transformer: training results</h1>
<p class="notice">TinyStories, 100 optimizer updates, context length 128.</p>
<p><strong>{parameters:,} parameters | Apple M4 / MPS / FP32 | 100 updates | 12,800 sampled tokens</strong></p>
<p>Fixed-window validation CE: {rows[0]["val_loss"]:.6f} → {rows[-1]["val_loss"]:.6f} nats/token.
Stopped at update 5, then resumed to update 100.</p>
<img src="../loss.svg" alt="Training and validation loss">
<p class="notice">Greedy completion:</p>
<pre>{html.escape(generation)}</pre></section>
<section id="audit"><h2>Checkpoint and optimizer inspection</h2>
<p>Training backend: MPS. Checkpoint inspection: CPU. <a href="audit.json">Inspection results</a></p>
<pre>.venv/bin/python -m scripts.evidence &gt; reports/checkpoint-verification.log</pre>
<pre>{html.escape(json.dumps(audit, indent=2))}</pre></section>
<section id="commands"><h2>Reproduction commands</h2><p>Working directory: <code>stages/01-basics</code>.
Existing nonempty run/data directories cannot be overwritten. <a href="commands.txt">Plain text commands</a></p>
<pre>{html.escape(COMMANDS)}</pre></section>
<section id="logs"><h2>Training log: updates 0, 5, 6 and 100</h2>
<p>JSON log view. Complete logs:
<a href="week1-training-first.log">updates 0–5</a> and <a href="week1-training-resume.log">updates 6–100</a>.</p>
<pre>{html.escape(selected)}</pre></section></main></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
