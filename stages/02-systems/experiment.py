"""Measure reference/SDPA attention and a hand-written KV cache on this machine."""

import argparse
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention

from aml.model import ModelConfig, TransformerLM
from aml.recording import save_record
from aml.train import select_device, synchronize


def reference_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    mask = torch.ones(q.shape[-2], k.shape[-2], device=q.device, dtype=torch.bool).triu(1)
    return torch.softmax(scores.masked_fill(mask, float("-inf")).float(), -1).to(q.dtype) @ v


def rotate(x: Tensor, offset: int) -> Tensor:
    length, width = x.shape[-2:]
    frequencies = 10000.0 ** (-torch.arange(0, width, 2, device=x.device).float() / width)
    angles = torch.arange(offset, offset + length, device=x.device).float()[:, None] * frequencies
    even, odd = x[..., ::2], x[..., 1::2]
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    return torch.stack((even*cos - odd*sin, even*sin + odd*cos), -1).flatten(-2)


class CachedDecoder:
    def __init__(self, model: TransformerLM):
        self.model = model.eval()
        self.layers: list[tuple[Tensor, Tensor]] = []
        self.length = 0

    @torch.no_grad()
    def step(self, ids: Tensor) -> Tensor:
        if ids.ndim != 2 or ids.shape[1] < 1:
            raise ValueError("Expected nonempty [batch, time] IDs")
        if self.length + ids.shape[1] > self.model.config.context_length:
            raise ValueError("Cache context exceeded; reset and prefill a new window")
        if self.layers and ids.shape[0] != self.layers[0][0].shape[0]:
            raise ValueError("Cache batch size changed")
        x = self.model.embedding[ids]
        batch, length, width = x.shape
        new_layers = []
        for index, block in enumerate(self.model.blocks):
            attention = block.attention
            q, k, v = attention.qkv(block.attn_norm(x)).chunk(3, -1)
            q, k, v = [
                value.reshape(batch, length, attention.n_heads, attention.head_dim).transpose(1, 2)
                for value in (q, k, v)
            ]
            q, k = rotate(q, self.length), rotate(k, self.length)
            if self.layers:
                old_k, old_v = self.layers[index]
                k, v = torch.cat((old_k, k), -2), torch.cat((old_v, v), -2)
            positions = torch.arange(self.length, self.length + length, device=x.device)
            blocked = torch.arange(k.shape[-2], device=x.device)[None, :] > positions[:, None]
            scores = q @ k.transpose(-1, -2) / math.sqrt(attention.head_dim)
            weights = torch.softmax(scores.masked_fill(blocked, float("-inf")).float(), -1).to(x.dtype)
            values = (weights @ v).transpose(1, 2).contiguous().reshape(batch, length, width)
            x = x + attention.output(values)
            x = x + block.ffn(block.ffn_norm(x))
            new_layers.append((k, v))
        self.layers = new_layers
        self.length += length
        return self.model.norm(x) @ self.model.embedding.T

    def bytes(self) -> int:
        return sum(value.numel() * value.element_size() for layer in self.layers for value in layer)


def measure(function, device: str, repeats: int = 7) -> dict:
    for _ in range(2):
        function()
    timings = []
    for _ in range(repeats):
        synchronize(device)
        began = time.perf_counter()
        function()
        synchronize(device)
        timings.append((time.perf_counter() - began) * 1000)
    return {"median_ms": statistics.median(timings), "min_ms": min(timings),
            "max_ms": max(timings), "repeats": repeats, "warmups": 2}


@torch.no_grad()
def decode(model: TransformerLM, prefix: Tensor, count: int, cached: bool) -> tuple[list[int], int]:
    decoder = CachedDecoder(model) if cached else None
    ids = prefix.clone()
    result = []
    for index in range(count):
        if decoder is None:
            logits = model(ids)
        else:
            logits = decoder.step(ids if index == 0 else ids[:, -1:])
        token = logits[:, -1].argmax(-1, keepdim=True)
        result.append(int(token.item()))
        ids = torch.cat((ids, token), 1)
    return result, decoder.bytes() if decoder else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    device = select_device(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    attention_results = []
    for length in (64, 128, 256):
        inputs = [torch.randn(1, 4, length, 16, device=device, requires_grad=True) for _ in range(3)]
        reference = reference_attention(*inputs)
        oracle = scaled_dot_product_attention(*inputs, is_causal=True)
        error = float((reference - oracle).abs().max().item())
        reference_grad = torch.autograd.grad(reference.square().mean(), inputs, retain_graph=True)
        oracle_grad = torch.autograd.grad(oracle.square().mean(), inputs)
        gradient_error = max(float((left-right).abs().max()) for left, right in zip(reference_grad, oracle_grad))
        torch.testing.assert_close(reference, oracle, atol=3e-5, rtol=3e-5)
        for actual, expected in zip(reference_grad, oracle_grad):
            torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
        def training_step(backend) -> None:
            values = [value.detach().requires_grad_() for value in inputs]
            backend(*values).square().mean().backward()
        old = measure(lambda: training_step(reference_attention), device)
        new = measure(lambda: training_step(lambda q, k, v: scaled_dot_product_attention(q, k, v, is_causal=True)), device)
        attention_results.append({"length": length, "forward_max_abs_error": error,
                                  "backward_max_abs_error": gradient_error,
                                  "reference_forward_backward": old, "torch_sdpa_forward_backward": new,
                                  "speedup": old["median_ms"] / new["median_ms"]})
    config = ModelConfig(vocab_size=512, context_length=128, d_model=64, n_layers=2, n_heads=4, d_ff=192)
    model = TransformerLM(config).to(device).eval()
    prefix = torch.randint(0, 512, (1, 32), device=device)
    generation = []
    for count in (16, 32, 64):
        plain, _ = decode(model, prefix, count, False)
        cached, memory = decode(model, prefix, count, True)
        if plain != cached:
            raise AssertionError("Greedy cache output differs from full-prefix decoding")
        old = measure(lambda: decode(model, prefix, count, False), device, 5)
        new = measure(lambda: decode(model, prefix, count, True), device, 5)
        generation.append({"prefix_tokens": 32, "new_tokens": count, "identical_output": True,
                           "uncached": old, "cached": new, "cache_bytes": memory,
                           "speedup": old["median_ms"] / new["median_ms"]})
    result = {
        "device": device, "precision": "float32", "model": asdict(config),
        "attention": attention_results, "generation": generation,
        "hardware_blocked": {"triton": "No compatible Triton GPU on current Mac",
                             "multi_gpu_training": "Only one Apple GPU; no multi-GPU measurement",
                             "multi_gpu_inference": "Only one Apple GPU; no multi-GPU measurement"},
        "protocol": "Isolated synchronized wall-clock timing, warmups excluded, medians and full ranges",
    }
    save_record(Path("reports"), "Stage 2 — measured attention and hand-written KV cache", result, {
        "1. Attention correctness and forward/backward timing": attention_results,
        "2. Identical greedy generation with/without KV cache": generation,
        "3. Device, configuration and hardware boundary": {key: value for key, value in result.items() if key not in ("attention", "generation")},
    })


if __name__ == "__main__":
    main()
