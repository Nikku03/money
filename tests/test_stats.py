import unittest

from brain.research.stats import bh_adjust, binom_lower_bound, binom_one_sided_greater, binom_two_sided


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

    def test_one_sided_binomial(self):
        self.assertAlmostEqual(binom_one_sided_greater(9, 10), 11 / 1024)
        self.assertAlmostEqual(binom_one_sided_greater(10, 10), 1 / 1024)
        self.assertEqual(binom_one_sided_greater(0, 10), 1.0)
        self.assertEqual(binom_one_sided_greater(11, 10), 0.0)
        self.assertAlmostEqual(binom_one_sided_greater(20, 30), 0.0493685733527, places=10)
        self.assertAlmostEqual(binom_one_sided_greater(2, 3, p=0.25), 10 / 64)
        big = binom_one_sided_greater(1100, 2000)  # past where math.comb fits a float
        self.assertTrue(0 < big < 1e-5)

    def test_clopper_pearson_lower_bound(self):
        # closed forms: k = n gives alpha ** (1/n); k = 1 gives 1 - (1 - alpha) ** (1/n)
        self.assertAlmostEqual(binom_lower_bound(10, 10), 0.05 ** 0.1, places=9)
        self.assertAlmostEqual(binom_lower_bound(1, 10), 1 - 0.95 ** 0.1, places=9)
        self.assertAlmostEqual(binom_lower_bound(30, 30, alpha=0.1), 0.1 ** (1 / 30), places=9)
        self.assertAlmostEqual(binom_lower_bound(7, 10), 0.3933758, places=6)  # Beta(7, 4) 5% quantile
        self.assertEqual(binom_lower_bound(0, 10), 0.0)
        self.assertEqual(binom_lower_bound(0, 0), 0.0)
        L = binom_lower_bound(27, 30)
        self.assertAlmostEqual(binom_one_sided_greater(27, 30, L), 0.05, places=9)
        self.assertGreater(L, 0.75)  # 27 of 30: about what a source needs to reach weight 0.5 alone
        self.assertLess(binom_lower_bound(26, 30), 0.75)
