import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from aml.generate import generate
from aml.model import ModelConfig, RMSNorm, TransformerLM, rope
from aml.tinystories import fetch_rows, prepare
from aml.tokenizer import ByteBPE


class Week1Tests(unittest.TestCase):
    def test_rope_rotation_and_rmsnorm(self):
        x = torch.randn(2, 3, 4, 8)
        rotated = rope(x)
        torch.testing.assert_close(rotated[:, :, 0], x[:, :, 0])
        torch.testing.assert_close(rotated.square().sum(-1), x.square().sum(-1))
        simple = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        expected = torch.tensor([[[[1.0, 0.0], [torch.cos(torch.tensor(1.0)), torch.sin(torch.tensor(1.0))]]]])
        torch.testing.assert_close(rope(simple), expected)
        norm = RMSNorm(8)
        expected_norm = x / (x.square().mean(-1, keepdim=True) + 1e-5).sqrt()
        torch.testing.assert_close(norm(x), expected_norm)

    def test_generation_limit_eos_and_invalid_temperature(self):
        model = TransformerLM(ModelConfig(vocab_size=257, context_length=4, d_model=8, n_layers=1, n_heads=1, d_ff=16))
        logits = torch.full((1, 1, 257), -100.0)
        logits[..., 97] = 100
        with patch.object(model, "forward", return_value=logits):
            self.assertEqual(generate(model, ByteBPE(), "x", 3, 0), "xaaa")
        logits[..., 256] = 200
        with patch.object(model, "forward", return_value=logits):
            self.assertEqual(generate(model, ByteBPE(), "x", 3, 0), "x")
        with self.assertRaises(ValueError):
            generate(model, ByteBPE(), "", 1, float("nan"))

    def test_download_validation(self):
        payload = {"rows": [{"row_idx": 0, "row": {"text": "one\nstory"}, "truncated_cells": []}]}
        with patch("aml.tinystories.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())):
            documents, pages = fetch_rows("train", 1)
        self.assertEqual(documents, ["one story"])
        self.assertEqual(len(pages[0]["sha256"]), 64)
        payload["rows"][0]["truncated_cells"] = ["text"]
        with patch("aml.tinystories.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())):
            with self.assertRaisesRegex(ValueError, "Truncated"):
                fetch_rows("train", 1)

    def test_subset_overlap_budget_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "raw"
            with patch("aml.tinystories.fetch_rows", side_effect=[(["train"], []), (["val"], [])]):
                manifest = prepare(output, 1, 1)
            self.assertEqual(manifest["license"], "CDLA-Sharing-1.0")
            self.assertEqual((output / "train.txt").read_text(), "train\n")
            with self.assertRaisesRegex(ValueError, "new or empty"):
                prepare(output, 1, 1)
            with self.assertRaisesRegex(ValueError, "budget"):
                prepare(output, 2001, 1)
            with patch("aml.tinystories.fetch_rows", return_value=(["same"], [])):
                with self.assertRaisesRegex(ValueError, "overlap"):
                    prepare(Path(directory) / "overlap", 1, 1)

    def test_walkthrough_covers_each_nonempty_source_line(self):
        for name in ("tokenizer", "data", "model", "train", "generate", "tinystories"):
            lines = Path(f"aml/{name}.py").read_text(encoding="utf-8").splitlines()
            guide = Path(f"docs/week1/{name}.md").read_text(encoding="utf-8")
            covered = {int(value) for value in re.findall(r"^\| (\d+) \|", guide, flags=re.MULTILINE)}
            expected = {index for index, line in enumerate(lines, 1) if line.strip()}
            self.assertEqual(covered, expected, f"Update {name} walkthrough after changing source")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
