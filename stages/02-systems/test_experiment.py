import unittest

import torch

from aml.model import ModelConfig, TransformerLM
from experiment import CachedDecoder, decode, reference_attention


class CacheTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(1)
        self.model = TransformerLM(ModelConfig(512, 16, 32, 2, 4, 64)).eval()

    def test_chunked_prefill_and_every_cached_logit_match(self):
        ids = torch.randint(0, 512, (2, 12))
        cache = CachedDecoder(self.model)
        with torch.no_grad():
            torch.testing.assert_close(cache.step(ids[:, :5]), self.model(ids[:, :5]), atol=1e-6, rtol=1e-5)
            for start, end in ((5, 8), (8, 9), (9, 12)):
                torch.testing.assert_close(cache.step(ids[:, start:end]),
                                           self.model(ids[:, :end])[:, start:end], atol=1e-6, rtol=1e-5)
        expected_bytes = 2 * 2 * 2 * 12 * 32 * 4
        self.assertEqual(cache.bytes(), expected_bytes)

    def test_cache_overflow_and_batch_change_are_explicit(self):
        cache = CachedDecoder(self.model)
        cache.step(torch.ones(1, 16, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "context"):
            cache.step(torch.ones(1, 1, dtype=torch.long))
        cache = CachedDecoder(self.model)
        cache.step(torch.ones(1, 1, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "batch"):
            cache.step(torch.ones(2, 1, dtype=torch.long))

    def test_greedy_equality(self):
        prompt = torch.tensor([[1, 2, 3]])
        self.assertEqual(decode(self.model, prompt, 8, True)[0], decode(self.model, prompt, 8, False)[0])

    def test_attention_gradients(self):
        values = [torch.randn(2, 4, 5, 8, requires_grad=True) for _ in range(3)]
        actual = reference_attention(*values)
        expected = torch.nn.functional.scaled_dot_product_attention(*values, is_causal=True)
        torch.testing.assert_close(actual, expected)
        for left, right in zip(torch.autograd.grad(actual.square().sum(), values),
                               torch.autograd.grad(expected.square().sum(), values)):
            torch.testing.assert_close(left, right, atol=3e-6, rtol=3e-5)


if __name__ == "__main__":
    unittest.main()
