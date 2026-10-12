import unittest

from brain.research.stats import bh_adjust, binom_two_sided


class StatsTest(unittest.TestCase):
    def test_binomial(self):
        self.assertAlmostEqual(binom_two_sided(0, 10), 2 / 1024)
        self.assertAlmostEqual(binom_two_sided(5, 10), 1.0)
        self.assertAlmostEqual(binom_two_sided(9, 10), 22 / 1024)

    def test_bh(self):
        q = bh_adjust([0.01, 0.04, 0.03, 0.5])
        self.assertAlmostEqual(q[0], 0.04)
        self.assertAlmostEqual(q[1], 0.04 * 4 / 3)
        self.assertAlmostEqual(q[2], 0.04 * 4 / 3)
        self.assertAlmostEqual(q[3], 0.5)
