"""Record tokenizer, data, model and training measurements for stage one."""

import json
import re
import subprocess
import sys
from pathlib import Path

import torch

from aml.data import sha256
from aml.generate import generate
from aml.model import ModelConfig, TransformerLM
from aml.recording import loss_svg, save_record
from aml.tokenizer import ByteBPE


def main() -> None:
    torch.set_num_threads(1)
    run = Path("runs/local-1500")
    report = Path("reports/local")
    tokenizer = ByteBPE.load(run / "tokenizer.json")
    train = Path("data/week1-raw/train.txt").read_text()
    validation = Path("data/week1-raw/val.txt").read_text()
    words = set(re.findall(r"\s+|[^\s]+", train))
    val_words = re.findall(r"\s+|[^\s]+", validation)
    sample = "The cat reads a book. 小猫在看书。"
    encoded = tokenizer.encode(sample)
    assert tokenizer.decode(encoded) == sample
    comparison = {
        "validation_characters": len(validation),
        "validation_bytes": len(validation.encode()),
        "character_tokens": len(validation),
        "word_tokens_including_whitespace": len(val_words),
        "word_unknown_pieces": sum(word not in words for word in val_words),
        "byte_bpe_tokens": len(tokenizer.encode(validation)),
        "byte_bpe_actual_vocabulary": tokenizer.vocab_size,
        "note": "Sequence-length comparison, not a comparison of LM losses.",
        "round_trip": {"text": sample, "ids": encoded, "decoded": tokenizer.decode(encoded)},
    }
    test = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        capture_output=True, text=True, check=True,
    )
    report.mkdir(parents=True, exist_ok=True)
    (report / "tests.log").write_text(test.stdout + test.stderr)
    state = torch.load(run / "latest.pt", map_location="cpu", weights_only=True)
    model = TransformerLM(ModelConfig(**state["config"]["model"])).to("mps")
    model.load_state_dict(state["model"])
    samples = {
        prompt: generate(model, tokenizer, prompt, 64, 0)
        for prompt in ("Once upon a time", "The little girl", "A dog")
    }
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    validation_points = [(row["step"], row["val_loss"]) for row in metrics if "val_loss" in row]
    loss_svg(validation_points, report / "loss.svg")
    result = {
        "parameters": model.parameter_count(), "updates": state["step"],
        "training_tokens": metrics[-1]["tokens_seen"],
        "initial_validation_ce": metrics[0]["val_loss"],
        "final_validation_ce": metrics[-1]["val_loss"],
        "evaluation_positions": 16 * 128,
        "training_step_seconds": sum(row.get("step_seconds", 0) for row in metrics),
        "checkpoint_sha256": sha256(run / "latest.pt"),
        "checkpoint_bytes": (run / "latest.pt").stat().st_size,
        "tokenizers": comparison, "generation": samples,
    }
    (report / "metrics.jsonl").write_bytes((run / "metrics.jsonl").read_bytes())
    (report / "manifest.json").write_bytes((run / "manifest.json").read_bytes())
    save_record(report, "Stage 1 — tokenizer, model, 98M training and generation", result, {
        "1. Tokenization on held-out stories": comparison,
        "2. Model and training measurements": {k: v for k, v in result.items() if k not in ("tokenizers", "generation")},
        "3. Greedy completions": samples,
        "4. Component-test output": test.stdout + test.stderr,
    })
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
