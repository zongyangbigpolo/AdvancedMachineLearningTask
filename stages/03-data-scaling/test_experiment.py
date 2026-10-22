import unittest

from experiment import clean, fit_power, shingles


class DataScalingTests(unittest.TestCase):
    def test_nested_templates_references_and_last_text(self):
        text = "Start {{a|{{b}}}} [[Cat|kitten]] <ref>citation</ref> end."
        self.assertEqual(clean(text), "Start kitten end.")

    def test_bad_markup_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "Unbalanced"):
            clean("A {{never closed")

    def test_shingle_identity(self):
        first = shingles("one two three four five six")
        self.assertEqual(len(first), 2)
        self.assertEqual(first, shingles("ONE two three four five six"))

    def test_power_fit_predicts_unseen_point(self):
        points = [(size, 1.0 + 3.0 * (size / 100) ** -0.5) for size in (100, 200, 400, 800)]
        result = fit_power(points)
        predicted = result["offset"] + result["amplitude"] * (1600 / result["n0"]) ** -result["alpha"]
        self.assertAlmostEqual(predicted, 1.75, delta=0.04)


if __name__ == "__main__":
    unittest.main()
