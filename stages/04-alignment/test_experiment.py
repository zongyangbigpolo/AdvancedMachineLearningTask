import unittest

import torch

from experiment import cases, dpo_loss, leave_one_out, verify


class AlignmentTests(unittest.TestCase):
    def test_disjoint_exhaustive_split(self):
        train, test = cases()
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(len(set(train) | set(test)), 100)

    def test_verifier_rejects_wrong_or_invalid_answers(self):
        self.assertEqual(verify((3, 7), 7), 1)
        self.assertEqual(verify((3, 7), 3), 0)
        with self.assertRaises(ValueError):
            verify((3, 7), 17)

    def test_dpo_identity_and_direction(self):
        reference = torch.tensor([[-1.0, -2.0]])
        chosen, rejected = torch.tensor([0]), torch.tensor([1])
        baseline = dpo_loss(reference, reference, chosen, rejected)
        self.assertAlmostEqual(float(baseline), float(torch.log(torch.tensor(2.0))))
        self.assertLess(dpo_loss(torch.tensor([[0.0, -3.0]]), reference, chosen, rejected), baseline)
        with self.assertRaises(ValueError):
            dpo_loss(reference, reference, chosen, rejected, 0)

    def test_leave_one_out_does_not_include_own_reward(self):
        rewards = torch.tensor([[1.0, 0.0, 0.0]])
        torch.testing.assert_close(leave_one_out(rewards), torch.tensor([[1.0, -0.5, -0.5]]))
        with self.assertRaises(ValueError):
            leave_one_out(torch.ones(2, 1))


if __name__ == "__main__":
    unittest.main()
