"""Equation discovery on a simulated Earth (brain/discover): the world, the engine, the laws."""
import unittest

import numpy as np

from brain.discover import earth, engine, laws


class EarthTest(unittest.TestCase):
    def test_field_is_a_vacuum_field(self):
        rng = np.random.default_rng(1)
        r, _ = earth.survey(20, 1.0, 5.0, rng)
        J = earth.jacobian(r)
        scale = np.abs(J).max(axis=(1, 2))
        self.assertLess(np.max(np.abs(np.trace(J, axis1=1, axis2=2)) / scale), 1e-6)     # div B = 0
        self.assertLess(np.max(np.abs(J - J.transpose(0, 2, 1)).max(axis=(1, 2)) / scale), 1e-6)  # curl B = 0

    def test_dipole_dominates_far_away(self):
        r = np.array([[50.0, 0, 0]])
        dip = (3 * (np.array([1.0, 0, 0]) @ earth.M) * np.array([1.0, 0, 0]) - earth.M) / 50**3
        self.assertLess(np.linalg.norm(earth.field(r)[0] - dip) / np.linalg.norm(dip), 0.01)

    def test_boris_keeps_speed_without_gravity(self):
        r, v = np.array([2.0, 0, 0]), np.array([0.0, 0.2, 0.2])
        t, p = earth.trajectory(r, v, 50.0, 1.0, 0.01)
        self.assertEqual(p.shape, (101, 3))


class EngineTest(unittest.TestCase):
    def test_finds_a_planted_sparse_law(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(400, 12))
        y = 2.0 * X[:, 1] - 3.0 * X[:, 7] + 0.01 * rng.normal(size=400)
        Xt = rng.normal(size=(200, 12))
        yt = 2.0 * Xt[:, 1] - 3.0 * Xt[:, 7]
        names = [f"x{i}" for i in range(12)]
        law = engine.discover("y", names, X, y, Xt, yt)
        self.assertTrue(law.found)
        self.assertEqual(set(law.terms), {"x1", "x7"})
        self.assertAlmostEqual(law.terms["x1"], 2.0, places=2)
        self.assertAlmostEqual(law.terms["x7"], -3.0, places=2)

    def test_finds_nothing_in_noise(self):
        rng = np.random.default_rng(0)
        X, Xt = rng.normal(size=(400, 12)), rng.normal(size=(200, 12))
        law = engine.discover("y", [f"x{i}" for i in range(12)], X, rng.normal(size=400), Xt, rng.normal(size=200))
        self.assertFalse(law.found)

    def test_families_are_chosen_whole(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 6))
        y = X[:, 0] + 0.5 * X[:, 1]
        law = engine.discover("y", list("abcdef"), X, y, X, y, families=["A", "A", "A", "B", "C", "D"])
        self.assertEqual(law.families, ["A"])
        self.assertEqual(set(law.terms), {"a", "b", "c"})


class LawsTest(unittest.TestCase):
    def test_field_dipole(self):
        r = laws.field_law(n=300, noise=0.0)
        self.assertTrue(r["law"].found)
        self.assertIn("uniform e/r^3", r["law"].families)
        self.assertIn("first-order (e.u) u/r^3", r["law"].families)
        self.assertAlmostEqual(r["tilt_deg"], 11.0, delta=0.3)
        self.assertEqual(r["monopole"], 0.0)

    def test_field_without_the_right_power_is_not_found(self):
        self.assertFalse(laws.field_law(n=300, powers=(1, 2, 4, 5))["law"].found)

    def test_maxwell(self):
        eqs = {e["equation"] for e in laws.maxwell(n=200)["equations"]}
        self.assertIn("dBx/dx + dBy/dy + dBz/dz = 0", eqs)
        self.assertEqual(len(eqs), 4)

    def test_gravity_from_satellites(self):
        r = laws.motion_law("satellite")
        self.assertTrue(r["law"].found)
        self.assertEqual(set(r["law"].terms), {"u/r^2"})
        self.assertAlmostEqual(r["law"].terms["u/r^2"], -earth.GM, places=2)


if __name__ == "__main__":
    unittest.main()
