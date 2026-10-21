"""Single-device FP32 reference training with verifiable data and resumable state."""

import argparse
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from aml.data import TokenStream, sha256
from aml.model import ModelConfig, TransformerLM, cross_entropy


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 20000
    batch_size: int = 2
    grad_accum: int = 16
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 200
    eval_batches: int = 20
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("steps", "batch_size", "grad_accum", "eval_interval", "eval_batches"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup_steps must be in [0, steps)")
        if not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if not all(math.isfinite(value) for value in (
            self.learning_rate, self.min_lr_ratio, self.weight_decay, self.grad_clip,
        )):
            raise ValueError("Training hyperparameters must be finite")
        if self.learning_rate <= 0 or self.grad_clip <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")


def learning_rate(step: int, config: TrainConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    decay_steps = config.steps - config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, decay_steps - 1)
    multiplier = config.min_lr_ratio + (1 - config.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
    return config.learning_rate * multiplier


def select_device(requested: str) -> str:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS requested but unavailable")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def device_rng(device: str):
    if device == "cuda":
        return torch.cuda.get_rng_state_all()
    if device == "mps":
        return torch.mps.get_rng_state()
    return None


def restore_device_rng(device: str, state) -> None:
    if device == "cuda":
        torch.cuda.set_rng_state_all(state)
    elif device == "mps":
        torch.mps.set_rng_state(state)


@torch.no_grad()
def evaluate(model: TransformerLM, stream: TokenStream, config: TrainConfig, device: str) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(config.seed + 1)
    losses = []
    for _ in range(config.eval_batches):
        x, y = stream.batch(config.batch_size, model.config.context_length, generator, device)
        losses.append(cross_entropy(model(x), y).item())
    model.train()
    loss = sum(losses) / len(losses)
    if not math.isfinite(loss):
        raise FloatingPointError("Non-finite validation loss")
    return loss


def verify_data(directory: Path, model: ModelConfig) -> str:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata["format"] != "uint32-le-v1":
        raise ValueError("Unsupported dataset format")
    if metadata["vocab_size"] > model.vocab_size:
        raise ValueError("Tokenizer vocabulary exceeds model vocabulary")
    if sha256(directory / "tokenizer.json") != metadata["tokenizer_sha256"]:
        raise ValueError("Tokenizer has changed since data preparation")
    for split in ("train", "val"):
        path = directory / f"{split}.bin"
        if path.stat().st_size != metadata[split]["tokens"] * 4:
            raise ValueError(f"{split} token count mismatch")
        if sha256(path) != metadata[split]["token_sha256"]:
            raise ValueError(f"{split} tokens have changed since data preparation")
        if metadata[split]["tokens"] <= model.context_length:
            raise ValueError(f"{split} is too short for context_length")
    return sha256(directory / "metadata.json")


def run(config_path: Path, data: Path, output: Path, device: str, resume: bool = False, stop_after: int | None = None) -> None:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    model_config, train_config = ModelConfig(**raw["model"]), TrainConfig(**raw["train"])
    device = select_device(device)
    end_step = train_config.steps if stop_after is None else stop_after
    if not 1 <= end_step <= train_config.steps:
        raise ValueError("stop_after must be in [1, configured steps]; it does not change the LR schedule")
    data_id = verify_data(data, model_config)
    checkpoint_path = output / "latest.pt"
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError("Output is not empty; choose a new run or use --resume")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True) if resume else None
    if state is not None:
        if state["config"] != raw or state["data_id"] != data_id or state["device"] != device:
            raise ValueError("Resume requires identical config, prepared data and device type")
        if not state["step"] < end_step:
            raise ValueError("Checkpoint already reached the requested stop step")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(train_config.seed)
    model = TransformerLM(model_config).to(device)
    optimizer = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": train_config.weight_decay},
         {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0}],
        lr=train_config.learning_rate, betas=(0.9, 0.95), eps=1e-8,
    )
    generator = torch.Generator().manual_seed(train_config.seed)
    start_step = 0
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["sampler_rng"])
        torch.set_rng_state(state["cpu_rng"])
        restore_device_rng(device, state["device_rng"])
        start_step = state["step"]
    else:
        manifest = {
            "config": raw, "data_id": data_id, "device": device,
            "parameters": model.parameter_count(), "python": platform.python_version(),
            "torch": str(torch.__version__), "platform": platform.platform(),
            "precision": "float32", "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name() if device == "cuda" else platform.machine(),
            "source_sha256": {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (output / "tokenizer.json").write_bytes((data / "tokenizer.json").read_bytes())
    log_path = output / "metrics.jsonl"
    if resume and log_path.exists():
        rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        log_path.write_text("".join(json.dumps(row) + "\n" for row in rows if row["step"] <= start_step), encoding="utf-8")
    train_stream, val_stream = TokenStream(data / "train.bin"), TokenStream(data / "val.bin")

    def record(metrics: dict) -> None:
        print(json.dumps(metrics), flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(metrics) + "\n")

    def save(step: int) -> None:
        checkpoint = {
            "step": step, "config": raw, "data_id": data_id, "device": device,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "sampler_rng": generator.get_state(), "cpu_rng": torch.get_rng_state(),
            "device_rng": device_rng(device),
            "tokenizer_sha256": sha256(data / "tokenizer.json"),
        }
        temporary = output / "latest.pt.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(checkpoint_path)

    try:
        if start_step == 0:
            record({"step": 0, "val_loss": evaluate(model, val_stream, train_config, device)})
        model.train()
        tokens_per_step = train_config.batch_size * model_config.context_length * train_config.grad_accum
        for step in range(start_step, end_step):
            synchronize(device)
            began = time.perf_counter()
            lr = learning_rate(step, train_config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for _ in range(train_config.grad_accum):
                x, y = train_stream.batch(train_config.batch_size, model_config.context_length, generator, device)
                loss = cross_entropy(model(x), y) / train_config.grad_accum
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at step {step + 1}")
                loss.backward()
                loss_sum += loss.item()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config.grad_clip, error_if_nonfinite=True,
            ).item()
            optimizer.step()
            synchronize(device)
            elapsed = time.perf_counter() - began
            metrics = {
                "step": step + 1, "train_loss": loss_sum, "lr": lr, "grad_norm": grad_norm,
                "tokens_seen": (step + 1) * tokens_per_step,
                "step_seconds": elapsed, "tokens_per_second": tokens_per_step / elapsed,
            }
            if (step + 1) % train_config.eval_interval == 0 or step + 1 == end_step:
                metrics["val_loss"] = evaluate(model, val_stream, train_config, device)
                save(step + 1)
            record(metrics)
    finally:
        train_stream.close()
        val_stream.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--threads", type=int)
    args = parser.parse_args()
    if args.threads is not None:
        if args.threads < 1:
            parser.error("--threads must be positive")
        torch.set_num_threads(args.threads)
    run(args.config, args.data, args.output, args.device, args.resume, args.stop_after)


if __name__ == "__main__":
    main()
