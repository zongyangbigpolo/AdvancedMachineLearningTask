"""Autoregressive reference generation; intentionally no KV cache yet."""

import argparse
import math
from pathlib import Path

import torch

from aml.data import sha256
from aml.model import ModelConfig, TransformerLM
from aml.tokenizer import ByteBPE
from aml.train import select_device


@torch.no_grad()
def generate(model: TransformerLM, tokenizer: ByteBPE, prompt: str, max_new_tokens: int, temperature: float) -> str:
    if max_new_tokens < 0 or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("max_new_tokens and finite temperature must be nonnegative")
    device = model.embedding.device
    ids = tokenizer.encode(prompt) or [tokenizer.eos_id]
    continuation = []
    model.eval()
    for _ in range(max_new_tokens):
        context = torch.tensor([ids[-model.config.context_length:]], dtype=torch.long, device=device)
        logits = model(context)[0, -1, :tokenizer.vocab_size]
        if temperature == 0:
            token = logits.argmax().item()
        else:
            probabilities = torch.softmax(logits.float() / temperature, dim=-1)
            token = torch.multinomial(probabilities, 1).item()
        if token == tokenizer.eos_id:
            break
        ids.append(token)
        continuation.append(token)
    return prompt + tokenizer.decode(continuation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 selects greedy decoding")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer_path = args.checkpoint.parent / "tokenizer.json"
    if sha256(tokenizer_path) != checkpoint["tokenizer_sha256"]:
        raise ValueError("Tokenizer does not match checkpoint")
    tokenizer = ByteBPE.load(tokenizer_path)
    model = TransformerLM(ModelConfig(**checkpoint["config"]["model"])).to(select_device(args.device))
    model.load_state_dict(checkpoint["model"])
    print(generate(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature))


if __name__ == "__main__":
    main()
