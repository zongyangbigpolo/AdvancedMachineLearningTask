"""Reload the trained neural agent, execute its tools, and print external verification."""

import argparse
import json
from pathlib import Path

import torch

from aml.data import sha256
from aml.model import ModelConfig, TransformerLM
from aml.tokenizer import ByteBPE
from experiment import LedgerEnvironment, NeuralPolicy, harness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/agent/agent.pt"))
    parser.add_argument("--self-judge", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    torch.set_num_threads(1)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer_path = args.checkpoint.parent / "tokenizer.json"
    if sha256(tokenizer_path) != state["tokenizer_sha256"]:
        raise ValueError("Tokenizer does not match this checkpoint")
    tokenizer = ByteBPE.load(tokenizer_path)
    model = TransformerLM(ModelConfig(**state["config"]["model"]))
    model.load_state_dict(state["model"])
    environment = LedgerEnvironment([1, 8, 4, 0, 9, 2, 5, 7], (6, 9, 5, 6, 2, 1, 6, 1))
    result = harness(environment, NeuralPolicy(model, tokenizer), args.self_judge, args.seed)
    for index, row in enumerate(result["trace"], 1):
        print(json.dumps({"step": index, **row}), flush=True)
    print(json.dumps({key: value for key, value in result.items() if key != "trace"}), flush=True)
    if not result["verified_success"]:
        raise SystemExit("External verifier rejected the final state")


if __name__ == "__main__":
    main()
