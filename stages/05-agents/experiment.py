"""Train and evaluate a neural tool-using agent in a resettable, verified environment."""

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.nn import functional as F

from aml.data import sha256
from aml.model import ModelConfig, TransformerLM
from aml.recording import save_record
from aml.tokenizer import ByteBPE


ACTIONS = ("L", "R", "S")


@dataclass
class LedgerEnvironment:
    values: list[int]
    targets: tuple[int, ...]
    budget: int = 128
    position: int = 0
    steps: int = 0
    done: bool = False
    reason: str = ""
    committed: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.values = list(self.values)
        if not self.targets or len(self.values) != len(self.targets) or self.budget < 1:
            raise ValueError("Need equally sized nonempty values/targets and a positive budget")
        if any(type(value) is not int or not 0 <= value <= 9 for value in (*self.values, *self.targets)):
            raise ValueError("Ledger values must be integers in 0..9")

    def observe(self) -> dict:
        if self.done:
            raise ValueError("Cannot observe a terminated episode")
        return {"current": self.values[self.position], "target": self.targets[self.position],
                "goal": self.position, "total_goals": len(self.targets)}

    def step(self, action: str) -> dict:
        if self.done:
            raise ValueError("Episode is already terminated")
        if action not in ACTIONS:
            raise ValueError("Unknown tool; allowed tools are L, R, S")
        self.steps += 1
        if action in ("L", "R"):
            value = self.values[self.position] + (-1 if action == "L" else 1)
            if not 0 <= value <= 9:
                self.done, self.reason = True, "range_violation"
            else:
                self.values[self.position] = value
        elif self.values[self.position] != self.targets[self.position]:
            self.done, self.reason = True, "incorrect_commit"
        else:
            self.committed.append(self.position)
            self.position += 1
            if self.position == len(self.targets):
                self.done, self.reason = True, "all_goals_committed"
        if self.steps >= self.budget and not self.done:
            self.done, self.reason = True, "step_budget_exhausted"
        return {"done": self.done, "reason": self.reason, "values": list(self.values)}

    def verified(self) -> bool:
        return (self.done and self.reason == "all_goals_committed"
                and self.position == len(self.targets)
                and self.committed == list(range(len(self.targets)))
                and tuple(self.values) == self.targets)


def teaching_action(current: int, target: int) -> str:
    return "L" if current > target else "R" if current < target else "S"


def policy_logits(model: TransformerLM, tokenizer: ByteBPE,
                  examples: list[tuple[int, int, str | None]]) -> torch.Tensor:
    prompts = [
        tokenizer.encode(f"cur:{current};target:{target};" +
                         ("move:" if draft is None else f"draft:{draft};review:"))
        for current, target, draft in examples
    ]
    if max(map(len, prompts)) > model.config.context_length:
        raise ValueError("Policy prompt exceeds the model context")
    ids = torch.full((len(prompts), max(map(len, prompts))), tokenizer.eos_id, dtype=torch.long)
    for row, prompt in enumerate(prompts):
        ids[row, :len(prompt)] = torch.tensor(prompt)
    logits = model(ids)[torch.arange(len(prompts)), torch.tensor([len(prompt)-1 for prompt in prompts])]
    return logits[:, torch.tensor([ord(action) for action in ACTIONS])]


class NeuralPolicy:
    def __init__(self, model: TransformerLM, tokenizer: ByteBPE):
        self.model, self.tokenizer = model.eval(), tokenizer
        self.calls = 0

    @torch.no_grad()
    def select(self, current: int, target: int, draft: str | None = None) -> str:
        self.calls += 1
        logits = policy_logits(self.model, self.tokenizer, [(current, target, draft)])
        return ACTIONS[int(logits.argmax(-1))]


def harness(environment: LedgerEnvironment, policy: NeuralPolicy, self_judge: bool,
            seed: int, draft_error_rate: float = 0) -> dict:
    if not 0 <= draft_error_rate <= 1:
        raise ValueError("Draft-error rate must lie in [0, 1]")
    rng = random.Random(seed)
    trace = []
    began = time.perf_counter()
    initial_calls = policy.calls
    while not environment.done:
        observation = environment.observe()
        draft = policy.select(observation["current"], observation["target"])
        if rng.random() < draft_error_rate:
            draft = rng.choice([action for action in ACTIONS if action != draft])
        action = (policy.select(observation["current"], observation["target"], draft)
                  if self_judge else draft)
        feedback = environment.step(action)
        trace.append({"observation": observation, "draft": draft, "action": action, "feedback": feedback})
    return {
        "verified_success": environment.verified(), "reason": environment.reason,
        "steps": environment.steps, "model_calls": policy.calls - initial_calls,
        "seconds": time.perf_counter() - began, "trace": trace,
        "final_values": list(environment.values), "targets": list(environment.targets),
    }


def demonstrations() -> list[tuple[int, int, str | None, int]]:
    rng = random.Random(2026)
    rows = []
    for length in (1, 2, 4, 8):
        for _ in range(40):
            env = LedgerEnvironment([rng.randrange(10) for _ in range(length)],
                                    tuple(rng.randrange(10) for _ in range(length)))
            while not env.done:
                observation = env.observe()
                current, target = observation["current"], observation["target"]
                action = teaching_action(current, target)
                rows.append((current, target, None, ACTIONS.index(action)))
                for draft in ACTIONS:
                    rows.append((current, target, draft, ACTIONS.index(action)))
                env.step(action)
            if not env.verified():
                raise AssertionError("Demonstration did not pass the external verifier")
    return rows


def evaluate(model: TransformerLM, tokenizer: ByteBPE, label: str,
             self_judge: bool = False, draft_error_rate: float = 0) -> tuple[list[dict], list[dict]]:
    policy = NeuralPolicy(model, tokenizer)
    results, examples = [], []
    for length in (1, 4, 8, 16):
        episodes = []
        for index in range(12):
            seed = 10000 + length * 100 + index
            rng = random.Random(seed)
            env = LedgerEnvironment([rng.randrange(10) for _ in range(length)],
                                    tuple(rng.randrange(10) for _ in range(length)), budget=128)
            measured = harness(env, policy, self_judge, seed + 1, draft_error_rate)
            episodes.append(measured)
            if index == 0:
                examples.append({"label": label, "goals": length, **measured})
        results.append({
            "label": label, "goals": length, "episodes": len(episodes),
            "verified_successes": sum(episode["verified_success"] for episode in episodes),
            "success_rate": sum(episode["verified_success"] for episode in episodes)/len(episodes),
            "mean_steps": sum(episode["steps"] for episode in episodes)/len(episodes),
            "mean_model_calls": sum(episode["model_calls"] for episode in episodes)/len(episodes),
            "total_seconds": sum(episode["seconds"] for episode in episodes),
            "draft_error_rate": draft_error_rate, "self_judge": self_judge,
        })
    return results, examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-steps", type=int, default=300)
    parser.add_argument("--rl-episodes", type=int, default=24)
    parser.add_argument("--base", type=Path, default=Path("../04-alignment/runs/alignment/sft-seed-42.pt"))
    parser.add_argument("--output", type=Path, default=Path("runs/agent"))
    args = parser.parse_args()
    if args.sft_steps < 1 or args.rl_episodes < 1:
        parser.error("Training budgets must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new --output directory; do not overwrite training")
    output.mkdir(parents=True)
    base = args.base
    state = torch.load(base, map_location="cpu", weights_only=True)
    tokenizer_path = base.parent / "tokenizer.json"
    if sha256(tokenizer_path) != state["tokenizer_sha256"]:
        raise ValueError("Agent base tokenizer mismatch")
    tokenizer = ByteBPE.load(tokenizer_path)
    model = TransformerLM(ModelConfig(**state["config"]["model"]))
    model.load_state_dict(state["model"])
    rows, traces = evaluate(model, tokenizer, "before_agent_training")
    dataset = demonstrations()
    (output / "demonstrations.json").write_text(json.dumps(dataset) + "\n")
    (output / "tokenizer.json").write_bytes(tokenizer_path.read_bytes())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    rng = random.Random(42)
    training = []
    began = time.perf_counter()
    model.train()
    for step in range(args.sft_steps):
        batch = rng.choices(dataset, k=32)
        logits = policy_logits(model, tokenizer, [row[:3] for row in batch])
        loss = F.cross_entropy(logits, torch.tensor([row[3] for row in batch]))
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite agent imitation loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % 25 == 0:
            training.append({"imitation_step": step + 1, "loss": float(loss.detach())})
    supervised_seconds = time.perf_counter() - began
    measured, examples = evaluate(model, tokenizer, "trajectory_sft")
    rows.extend(measured)
    traces.extend(examples)
    torch.save({"config": state["config"], "model": model.state_dict(),
                "tokenizer_sha256": state["tokenizer_sha256"]}, output / "trajectory-sft.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    model.train()
    began = time.perf_counter()
    for episode in range(args.rl_episodes):
        goals = min(4, 1 + episode // 8)
        env = LedgerEnvironment([rng.randrange(10) for _ in range(goals)],
                                tuple(rng.randrange(10) for _ in range(goals)), budget=64)
        log_probabilities, entropies = [], []
        while not env.done:
            observation = env.observe()
            logits = policy_logits(model, tokenizer, [(observation["current"], observation["target"], None)])
            log_probs = logits.log_softmax(-1)[0]
            action = torch.multinomial(log_probs.exp(), 1).item()
            log_probabilities.append(log_probs[action])
            entropies.append(-(log_probs.exp() * log_probs).sum())
            env.step(ACTIONS[action])
        reward = float(env.verified())
        loss = -(reward - 0.5) * torch.stack(log_probabilities).sum() - 0.001 * torch.stack(entropies).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite agent policy-gradient loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        training.append({"rl_episode": episode + 1, "goals": goals, "steps": env.steps,
                         "externally_verified_reward": reward, "loss": float(loss.detach())})
    reinforcement_seconds = time.perf_counter() - began
    for label, judge, noise in (("trajectory_sft_plus_rl", False, 0),
                                ("plus_self_judge", True, 0),
                                ("draft_noise_without_judge", False, 0.1),
                                ("draft_noise_with_judge", True, 0.1)):
        measured, examples = evaluate(model, tokenizer, label, judge, noise)
        rows.extend(measured)
        traces.extend(examples)
    checkpoint = output / "agent.pt"
    torch.save({"config": state["config"], "model": model.state_dict(),
                "tokenizer_sha256": state["tokenizer_sha256"], "base_sha256": sha256(base)}, checkpoint)
    result = {
        "environment": "Sequential integer-ledger goals. L decrements, R increments, S commits current goal.",
        "policy": "Transformer logits over L/R/S. No teaching_action call in inference or self-judge.",
        "verifier": "Final values, ordered commits and terminated-success state must agree.",
        "training": {"base_sha256": sha256(base), "parameters": model.parameter_count(),
                     "demonstration_rows": len(dataset), "imitation_updates": args.sft_steps,
                     "rl_episodes": args.rl_episodes, "imitation_seconds": supervised_seconds,
                     "reinforcement_seconds": reinforcement_seconds, "curve": training},
        "evaluation": rows, "checkpoint_sha256": sha256(checkpoint),
        "self_judge": "Same model, separately trained review prompt, no access to external verifier during correction.",
        "split": "Training and test use different episode seeds; test goal sequences are new, digit domain is shared.",
        "stress_test": "10% draft-action corruption is a separate declared robustness test, not ordinary performance.",
    }
    report = Path("reports")
    save_record(report, "Stage 5 — trained agent, external verification and self-judge", result, {
        "1. Imitation and environment-reward training": result["training"],
        "2. Held-out episode success and inference costs": rows,
        "3. Replayable neural-agent trajectory (selected transitions)": {
            "episode": {key: value for key, value in traces[-2].items() if key != "trace"},
            "first_transitions": traces[-2]["trace"][:4],
            "last_transitions": traces[-2]["trace"][-4:],
        },
    })
    (report / "traces.json").write_text(json.dumps(traces, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
