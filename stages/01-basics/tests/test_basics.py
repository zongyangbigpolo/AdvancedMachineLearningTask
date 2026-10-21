import contextlib
import io
import json
import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from aml.data import TokenStream, encode_file, sha256
from aml.generate import generate
from aml.model import Attention, ModelConfig, TransformerLM, cross_entropy, rope
from aml.tokenizer import ByteBPE, merge_pair
from aml.train import TrainConfig, learning_rate, run, verify_data


class TokenizerTests(unittest.TestCase):
    def test_round_trip_and_eos(self):
        texts = ["", "你好，世界！🙂", "hello  world\n\t42_!", "e\u0301 café\x00", "<eos>", "１２３ αβ"]
        tokenizer = ByteBPE.train(texts + ["hello hello hello"], 300)
        for text in texts:
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
            ids = tokenizer.encode(text, add_eos=True)
            self.assertEqual(ids[-1], 256)
            self.assertEqual(tokenizer.decode(ids), text)
        self.assertNotIn(256, tokenizer.encode("<eos>"))

    def test_determinism_and_serialization(self):
        tokenizer = ByteBPE.train(["banana bandana 中文"], 280)
        self.assertEqual(tokenizer.merges, ByteBPE.train(["banana bandana 中文"], 280).merges)
        self.assertLess(len(tokenizer.encode("banana")), len("banana".encode()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bpe.json"
            tokenizer.save(path)
            self.assertEqual(tokenizer.encode("banana 中文"), ByteBPE.load(path).encode("banana 中文"))

    def test_nonoverlapping_merge(self):
        self.assertEqual(merge_pair((1, 1, 1), (1, 1), 257), (257, 1))
        with self.assertRaises(ValueError):
            ByteBPE.train([], 270)
        with self.assertRaises(ValueError):
            ByteBPE.train(["text"], 256)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = ModelConfig(vocab_size=300, context_length=16, d_model=32, n_layers=2, n_heads=4, d_ff=64)

    def test_attention_matches_torch_oracle(self):
        attention = Attention(self.config)
        x = torch.randn(2, 7, 32, requires_grad=True)
        q, k, v = [
            value.reshape(2, 7, 4, 8).transpose(1, 2)
            for value in attention.qkv(x).chunk(3, dim=-1)
        ]
        reference = F.scaled_dot_product_attention(rope(q), rope(k), v, is_causal=True)
        reference = attention.output(reference.transpose(1, 2).reshape(2, 7, 32))
        actual = attention(x)
        torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)
        actual_gradient = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
        reference_gradient = torch.autograd.grad(reference.square().sum(), x)[0]
        torch.testing.assert_close(actual_gradient, reference_gradient, atol=2e-5, rtol=1e-4)

    def test_causality_and_gradients(self):
        model = TransformerLM(self.config)
        ids = torch.randint(300, (2, 10))
        changed = ids.clone()
        changed[:, 6:] = (changed[:, 6:] + 1) % 300
        torch.testing.assert_close(model(ids)[:, :6], model(changed)[:, :6], atol=1e-6, rtol=1e-5)
        cross_entropy(model(ids), ids).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 17, dtype=torch.long))

    def test_cross_entropy_matches_oracle(self):
        logits = (torch.randn(3, 5, 17) * 100).requires_grad_()
        targets = torch.randint(17, (3, 5))
        actual = cross_entropy(logits, targets)
        expected = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, logits, retain_graph=True)[0],
            torch.autograd.grad(expected, logits)[0],
        )

    def test_overfit_fixed_batch(self):
        model = TransformerLM(self.config)
        x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        y = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
        initial = cross_entropy(model(x), y).item()
        for _ in range(40):
            optimizer.zero_grad()
            loss = cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
        self.assertLess(cross_entropy(model(x), y).item(), initial * 0.1)
        self.assertEqual(generate(model, ByteBPE(), "abc", 0, 0), "abc")

    def test_100m_parameter_count(self):
        config = ModelConfig(**json.loads(Path("configs/100m.json").read_text())["model"])
        with torch.device("meta"):
            model = TransformerLM(config)
        expected = config.vocab_size * config.d_model + config.n_layers * (
            4 * config.d_model**2 + 3 * config.d_model * config.d_ff + 2 * config.d_model
        ) + config.d_model
        self.assertEqual(model.parameter_count(), expected)
        self.assertEqual(model.parameter_count(), 98_323_200)
        self.assertTrue(95_000_000 <= model.parameter_count() <= 105_000_000)


class TrainingTests(unittest.TestCase):
    def make_data(self, root):
        data = root / "data"
        data.mkdir()
        tokenizer = ByteBPE.train(["the cat walks in the garden"], 280)
        tokenizer.save(data / "tokenizer.json")
        metadata = {
            "format": "uint32-le-v1", "vocab_size": tokenizer.vocab_size,
            "eos_id": tokenizer.eos_id, "tokenizer_sha256": sha256(data / "tokenizer.json"),
        }
        for split, text in (("train", "the cat walks in the garden\n"), ("val", "the cat reads in the garden\n")):
            source = root / f"{split}.txt"
            source.write_text(text * 12, encoding="utf-8")
            metadata[split] = encode_file(source, data / f"{split}.bin", tokenizer)
        (data / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return data

    def test_resume_exactly_matches_uninterrupted_cpu(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            data = self.make_data(root)
            raw = {
                "model": asdict(ModelConfig(vocab_size=300, context_length=8, d_model=16, n_layers=1, n_heads=2, d_ff=32)),
                "train": asdict(TrainConfig(steps=8, batch_size=2, grad_accum=2, warmup_steps=2, eval_interval=4, eval_batches=2)),
            }
            config = root / "config.json"
            config.write_text(json.dumps(raw), encoding="utf-8")
            run(config, data, root / "full", "cpu")
            run(config, data, root / "resumed", "cpu", stop_after=3)
            run(config, data, root / "resumed", "cpu", resume=True)
            full = torch.load(root / "full/latest.pt", weights_only=True)
            resumed = torch.load(root / "resumed/latest.pt", weights_only=True)
            self.assertEqual(resumed["step"], 8)
            for key in full["model"]:
                torch.testing.assert_close(full["model"][key], resumed["model"][key], rtol=0, atol=0)
            torch.testing.assert_close(full["sampler_rng"], resumed["sampler_rng"], rtol=0, atol=0)
            full_rows = [json.loads(line) for line in (root / "full/metrics.jsonl").read_text().splitlines()]
            rows = [json.loads(line) for line in (root / "resumed/metrics.jsonl").read_text().splitlines()]
            self.assertEqual([row["step"] for row in rows], list(range(9)))
            self.assertEqual(full_rows[-1]["val_loss"], rows[-1]["val_loss"])
            self.assertTrue(all(math.isfinite(row["train_loss"]) for row in rows[1:]))
            with self.assertRaisesRegex(ValueError, "not empty"):
                run(config, data, root / "full", "cpu")
            raw["train"]["learning_rate"] *= 2
            config.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "identical config"):
                run(config, data, root / "resumed", "cpu", resume=True)

    def test_data_shift_and_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.make_data(Path(directory))
            stream = TokenStream(data / "train.bin")
            try:
                x, y = stream.batch(2, 8, torch.Generator().manual_seed(1), "cpu")
                torch.testing.assert_close(x[:, 1:], y[:, :-1])
            finally:
                stream.close()
            config = ModelConfig(context_length=8)
            verify_data(data, config)
            with (data / "train.bin").open("r+b") as stream:
                stream.write(b"\xff")
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_data(data, config)

    def test_schedule_and_invalid_config(self):
        config = TrainConfig(steps=10, warmup_steps=2)
        self.assertEqual(learning_rate(0, config), config.learning_rate / 2)
        self.assertEqual(learning_rate(1, config), config.learning_rate)
        self.assertAlmostEqual(learning_rate(9, config), config.learning_rate * config.min_lr_ratio)
        with self.assertRaises(ValueError):
            TrainConfig(grad_accum=0)
        with self.assertRaises(ValueError):
            ModelConfig(d_model=32, n_heads=3)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
