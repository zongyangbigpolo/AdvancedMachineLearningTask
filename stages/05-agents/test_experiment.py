import unittest
from unittest.mock import patch

import torch

from aml.model import ModelConfig, TransformerLM
from aml.tokenizer import ByteBPE
from experiment import LedgerEnvironment, NeuralPolicy, demonstrations, harness


class AgentTests(unittest.TestCase):
    def test_ordered_goal_verification_and_reset(self):
        env = LedgerEnvironment([0, 2], (1, 1))
        for action in ("R", "S", "L", "S"):
            env.step(action)
        self.assertTrue(env.verified())
        self.assertEqual(env.committed, [0, 1])
        with self.assertRaises(ValueError):
            env.step("S")
        self.assertEqual(LedgerEnvironment([0, 2], (1, 1)).position, 0)

    def test_invalid_tool_range_wrong_commit_and_budget(self):
        with self.assertRaises(ValueError):
            LedgerEnvironment([1], (1,)).step("shell")
        env = LedgerEnvironment([0], (1,))
        env.step("L")
        self.assertEqual(env.reason, "range_violation")
        self.assertFalse(env.verified())
        env = LedgerEnvironment([0], (1,))
        env.step("S")
        self.assertEqual(env.reason, "incorrect_commit")
        env = LedgerEnvironment([0], (2,), budget=1)
        env.step("R")
        self.assertEqual(env.reason, "step_budget_exhausted")
        self.assertFalse(env.verified())

    def test_neural_harness_does_not_call_demonstration_oracle(self):
        torch.set_num_threads(1)
        model = TransformerLM(ModelConfig(257, 64, 32, 1, 4, 64))
        policy = NeuralPolicy(model, ByteBPE())
        with patch("experiment.teaching_action", side_effect=AssertionError("Oracle used at inference")):
            result = harness(LedgerEnvironment([0], (1,), budget=3), policy, True, 4)
        self.assertGreater(result["model_calls"], 0)
        self.assertEqual(result["model_calls"], result["steps"] * 2)

    def test_demonstrations_and_review_signals(self):
        data = demonstrations()
        self.assertGreater(len(data), 100)
        self.assertTrue(any(row[2] is None for row in data))
        self.assertTrue(any(row[2] == "L" for row in data))


if __name__ == "__main__":
    unittest.main()
