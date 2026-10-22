"""Train SFT, DPO and verifiable-reward policy gradients from one frozen base."""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from aml.data import sha256
from aml.model import ModelConfig, TransformerLM
from aml.recording import save_record
from aml.tokenizer import ByteBPE


def cases() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    pairs = [(left, right) for left in range(10) for right in range(10)]
    random.Random(2026).shuffle(pairs)
    return pairs[:80], pairs[80:]


def action_logits(model: TransformerLM, tokenizer: ByteBPE, pairs: list[tuple[int, int]]) -> torch.Tensor:
    prompts = [tokenizer.encode(f"max({left},{right})=") for left, right in pairs]
    ids = torch.full((len(prompts), max(map(len, prompts))), tokenizer.eos_id, dtype=torch.long)
    for index, prompt in enumerate(prompts):
        ids[index, :len(prompt)] = torch.tensor(prompt)
    logits = model(ids)[torch.arange(len(prompts)), torch.tensor([len(prompt)-1 for prompt in prompts])]
    digit_ids = torch.tensor([ord(str(digit)) for digit in range(10)])
    return logits[:, digit_ids]


def verify(pair: tuple[int, int], answer: int) -> float:
    if not isinstance(answer, int) or not 0 <= answer <= 9:
        raise ValueError("Answer must be a single legal digit")
    return float(answer == max(pair))


def dpo_loss(policy: torch.Tensor, reference: torch.Tensor, chosen: torch.Tensor,
             rejected: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
    if beta <= 0:
        raise ValueError("DPO beta must be positive")
    rows = torch.arange(len(chosen))
    relative = (policy[rows, chosen] - policy[rows, rejected]
                - reference[rows, chosen] + reference[rows, rejected])
    return F.softplus(-beta * relative).mean()


def leave_one_out(rewards: torch.Tensor) -> torch.Tensor:
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("Need at least two rollout rewards per prompt")
    baseline = (rewards.sum(-1, keepdim=True) - rewards) / (rewards.shape[-1] - 1)
    return rewards - baseline


@torch.no_grad()
def score(model: TransformerLM, tokenizer: ByteBPE, pairs: list[tuple[int, int]]) -> dict:
    model.eval()
    predictions = action_logits(model, tokenizer, pairs).argmax(-1).tolist()
    result = {
        "correct": sum(verify(pair, answer) for pair, answer in zip(pairs, predictions)),
        "total": len(pairs),
        "examples": [{"prompt": f"max({pair[0]},{pair[1]})=", "answer": answer,
                      "expected": max(pair), "correct": bool(verify(pair, answer))}
                     for pair, answer in zip(pairs, predictions)],
    }
    result["accuracy"] = result["correct"] / result["total"]
    model.train()
    return result


def train(method: str, seed: int, steps: int, base: Path, output: Path) -> dict:
    state = torch.load(base, map_location="cpu", weights_only=True)
    tokenizer = ByteBPE.load(base.parent / "tokenizer.json")
    if sha256(base.parent / "tokenizer.json") != state["tokenizer_sha256"]:
        raise ValueError("Base tokenizer changed")
    torch.manual_seed(seed)
    config = ModelConfig(**state["config"]["model"])
    model = TransformerLM(config)
    model.load_state_dict(state["model"])
    reference = TransformerLM(config).eval()
    reference.load_state_dict(state["model"])
    reference.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    training, testing = cases()
    baseline = score(model, tokenizer, testing)
    rng = random.Random(seed)
    curve = []
    rewarded_rollouts = 0
    generated_rollouts = 0
    began = time.perf_counter()
    for step in range(steps):
        batch = rng.choices(training, k=32)
        targets = torch.tensor([max(pair) for pair in batch])
        log_policy = action_logits(model, tokenizer, batch).log_softmax(-1)
        if method == "sft":
            loss = F.nll_loss(log_policy, targets)
        elif method == "dpo":
            rejected = torch.tensor([rng.choice([value for value in range(10) if value != target])
                                     for target in targets.tolist()])
            with torch.no_grad():
                log_reference = action_logits(reference, tokenizer, batch).log_softmax(-1)
            loss = dpo_loss(log_policy, log_reference, targets, rejected)
        elif method == "rlvr":
            with torch.no_grad():
                actions = torch.multinomial(log_policy.exp(), 8, replacement=True)
                rewards = torch.tensor([[verify(pair, int(action)) for action in row]
                                        for pair, row in zip(batch, actions.tolist())])
                log_reference = action_logits(reference, tokenizer, batch).log_softmax(-1)
            advantages = leave_one_out(rewards)
            kl = (log_policy.exp() * (log_policy - log_reference)).sum(-1).mean()
            loss = -(log_policy.gather(1, actions) * advantages).mean() + 0.02 * kl
            generated_rollouts += rewards.numel()
            rewarded_rollouts += int(rewards.sum())
        else:
            raise ValueError(f"Unknown method: {method}")
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {method} objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % 25 == 0:
            curve.append({"step": step + 1, "training_objective": float(loss.detach())})
    measured_seconds = time.perf_counter() - began
    training_score = score(model, tokenizer, training)
    test_score = score(model, tokenizer, testing)
    checkpoint = output / f"{method}-seed-{seed}.pt"
    torch.save({**{key: state[key] for key in ("config", "tokenizer_sha256", "device")},
                "model": model.state_dict(), "step": steps, "base_sha256": sha256(base),
                "method": method, "seed": seed}, checkpoint)
    return {
        "method": method, "seed": seed, "base_sha256": sha256(base),
        "parameters": model.parameter_count(), "updates": steps, "prompts_seen": steps*32,
        "base_test_accuracy": baseline["accuracy"], "train_accuracy": training_score["accuracy"],
        "test": test_score, "seconds": measured_seconds, "curve": curve,
        "generated_rollouts": generated_rollouts, "correct_rollouts": rewarded_rollouts,
        "checkpoint_sha256": sha256(checkpoint),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--base", type=Path, default=Path("../03-data-scaling/runs/scaling/holdout.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/alignment"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    torch.set_num_threads(1)
    base = args.base
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new --output directory; do not overwrite an experiment")
    output.mkdir(parents=True)
    (output / "tokenizer.json").write_bytes((base.parent / "tokenizer.json").read_bytes())
    runs = []
    for method in ("sft", "dpo", "rlvr"):
        for seed in (42, 43, 44):
            result = train(method, seed, args.steps, base, output)
            runs.append(result)
            print(json.dumps(result), flush=True)
    training, testing = cases()
    comparison = []
    for method in ("sft", "dpo", "rlvr"):
        selected = [run for run in runs if run["method"] == method]
        comparison.append({
            "method": method, "mean_test_accuracy": sum(run["test"]["accuracy"] for run in selected)/3,
            "seed_test_accuracies": [run["test"]["accuracy"] for run in selected],
            "mean_training_seconds": sum(run["seconds"] for run in selected)/3,
            "updates_per_seed": args.steps, "prompts_per_seed": args.steps*32,
            "generated_rollouts_per_seed": selected[0]["generated_rollouts"],
        })
    result = {
        "task": "max(a,b), integers 0..9; single digit answer",
        "policy": "The same 10-digit candidate mask is used for all methods and the base",
        "base_sha256": sha256(base), "train_pairs": training, "test_pairs": testing,
        "protocol": "Same frozen base, no SFT warm-start for DPO/RLVR; equal update/prompt budgets, measured unequal wall time",
        "sft": "Answer-only negative log likelihood",
        "dpo": "Reference-relative preference objective; beta=0.5; correct vs uniformly sampled wrong digit",
        "rlvr": "On-policy REINFORCE, 8 one-token actions/prompt, leave-one-out baseline, exact verifier, KL coefficient 0.02",
        "comparison": comparison, "runs": runs,
    }
    save_record(Path("reports"), "Stage 4 — same-base SFT / DPO / RLVR measurements", result, {
        "1. Fixed task, data split and starting model": {key: value for key, value in result.items() if key not in ("runs", "comparison")},
        "2. Actual three-seed comparison and training costs": comparison,
        "3. Held-out answers (seed 42, including errors)": {
            run["method"]: run["test"] for run in runs if run["seed"] == 42
        },
    })


if __name__ == "__main__":
    main()
