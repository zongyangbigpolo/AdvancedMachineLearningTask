"""Pre-norm decoder: RMSNorm, RoPE, causal attention and SwiGLU."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 8192
    context_length: int = 512
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 2304

    def __post_init__(self) -> None:
        if any(value < 1 for value in vars(self).values()):
            raise ValueError("All model dimensions must be positive")
        if self.vocab_size < 257:
            raise ValueError("Vocabulary must include byte tokens and EOS")
        if self.d_model % self.n_heads or (self.d_model // self.n_heads) % 2:
            raise ValueError("d_model / n_heads must be an even integer for RoPE")


class Linear(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_out, n_in))
        nn.init.normal_(self.weight, std=n_in**-0.5)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T


class RMSNorm(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + 1e-5).to(x.dtype) * self.weight


def rope(x: Tensor) -> Tensor:
    length, width = x.shape[-2:]
    frequencies = 10000.0 ** (-torch.arange(0, width, 2, device=x.device).float() / width)
    angles = torch.arange(length, device=x.device).float()[:, None] * frequencies[None, :]
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = Linear(config.d_model, 3 * config.d_model)
        self.output = Linear(config.d_model, config.d_model)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2) for t in (q, k, v)]
        q, k = rope(q), rope(k)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        causal_mask = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        weights = torch.softmax(scores.float(), dim=-1).to(x.dtype)
        values = (weights @ v).transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.output(values)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate = Linear(config.d_model, config.d_ff)
        self.up = Linear(config.d_model, config.d_ff)
        self.down = Linear(config.d_ff, config.d_model)

    def forward(self, x: Tensor) -> Tensor:
        gate = self.gate(x)
        return self.down((gate * torch.sigmoid(gate)) * self.up(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Parameter(torch.empty(config.vocab_size, config.d_model))
        nn.init.normal_(self.embedding, std=0.02)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        for block in self.blocks:
            for projection in (block.attention.output, block.ffn.down):
                with torch.no_grad():
                    projection.weight.mul_((2 * config.n_layers) ** -0.5)

    def forward(self, ids: Tensor) -> Tensor:
        if ids.ndim != 2 or not 0 < ids.shape[1] <= self.config.context_length:
            raise ValueError("Expected [batch, time] within context_length")
        x = self.embedding[ids]
        for block in self.blocks:
            x = block(x)
        return self.norm(x) @ self.embedding.T

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    logits = logits.float()
    shifted = logits - logits.amax(dim=-1, keepdim=True)
    log_probs = shifted - shifted.exp().sum(dim=-1, keepdim=True).log()
    return -log_probs.gather(-1, targets.unsqueeze(-1)).mean()
