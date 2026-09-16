import math
import unittest

import numpy as np

from tpmo import ConstraintSpec, Evaluation, acceptable_margin, normalized_violation


class CoreTests(unittest.TestCase):
    def test_margin_decays(self):
        self.assertLess(acceptable_margin(10, 0.1), acceptable_margin(0, 0.1))

    def test_upper_constraint_relaxes(self):
        specs = [ConstraintSpec("c", "le", 10.0)]
        v = normalized_violation(np.array([[10.5]]), specs, 0.1)
        self.assertEqual(float(v[0, 0]), 0.0)
        v = normalized_violation(np.array([[11.5]]), specs, 0.1)
        self.assertGreater(float(v[0, 0]), 0.0)

    def test_lower_constraint_relaxes(self):
        specs = [ConstraintSpec("c", "ge", 10.0)]
        v = normalized_violation(np.array([[9.5]]), specs, 0.1)
        self.assertEqual(float(v[0, 0]), 0.0)
        v = normalized_violation(np.array([[8.5]]), specs, 0.1)
        self.assertGreater(float(v[0, 0]), 0.0)

    def test_evaluation_container(self):
        e = Evaluation(1.25, np.array([0.0]))
        self.assertTrue(math.isfinite(e.objective))
        self.assertEqual(e.constraints.shape, (1,))


if __name__ == "__main__":
    unittest.main()
