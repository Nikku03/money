"""The ladder of cells (LIF, Izhikevich, Hodgkin-Huxley), the exchange rate and the timing task."""
from __future__ import annotations

import unittest

import numpy as np

from brain.neurons import circuit, exchange, pong, train


def one(cells, kind="RS"):
    c = circuit.Circuit()
    c.add("X", kind)
    return c.compile(0, cells)


class Cells(unittest.TestCase):
    def test_izhikevich_is_the_default(self):
        c = circuit.Circuit()
        c.add("X")
        self.assertEqual(c.compile(0).cells, "izh")
        self.assertEqual(circuit.STATE_VARIABLES, {"lif": 1, "izh": 2, "hh": 4})

    def test_unknown_cells_refused(self):
        with self.assertRaises(ValueError):
            one("abc")

    def test_every_cell_rests_without_input(self):
        for cells in circuit.CELLS:
            s = one(cells)
            n = sum(len(s.step()) for _ in range(int(300 / s.dt)))
            self.assertEqual(n, 0, cells)

    def test_hh_rests_near_minus_65_and_fires_on_strong_drive(self):
        s = one("hh")
        for _ in range(400):
            s.step()
        self.assertAlmostEqual(float(s.v[0]), -65.0, delta=0.5)
        b = np.array([12.0 / s.gcur])               # 12 uA/cm2, well above rheobase
        n = sum(len(s.step(b)) for _ in range(4000))
        self.assertGreater(n, 50)                    # more than 50 spikes/s
        self.assertLess(n, 200)

    def test_hh_integrator_matches_a_fine_reference(self):
        times = {}
        for sub in (circuit.HH_SUBSTEPS, 50):
            old = circuit.HH_SUBSTEPS
            circuit.HH_SUBSTEPS = sub
            try:
                s = one("hh")
                b = np.array([10.0 / s.gcur])
                times[sub] = [s.t for _ in range(400) if len(s.step(b))]
            finally:
                circuit.HH_SUBSTEPS = old
        a, b = times[circuit.HH_SUBSTEPS], times[50]
        self.assertEqual(len(a), len(b))
        self.assertLessEqual(max(abs(x - y) for x, y in zip(a, b)), 0.25)

    def test_hh_rebound_after_inhibition(self):
        self.assertTrue(exchange.rebound("hh", None))

    def test_lif_subthreshold_is_exact(self):
        s = one("lif")
        ext = 10.0                                   # below LIF_THRESH - LIF_REST = 20
        b = np.array([ext / s.gcur])
        for _ in range(4):
            s.step(b)
        t = 4 * s.dt
        want = circuit.LIF_REST + ext * (1 - np.exp(-t / circuit.LIF_TAU))
        self.assertAlmostEqual(float(s.v[0]), want, places=6)

    def test_calibration_holds(self):
        w = circuit.threshold(lambda x: circuit.fires_on_event(x, "izh"), 0.0, 2.0)
        a = circuit.threshold(lambda x: circuit.fires_on_pulse(x, "izh"), 0.0, 400.0)
        for cells in ("hh", "lif"):
            self.assertTrue(circuit.fires_on_event(w * 1.03, cells), cells)
            self.assertFalse(circuit.fires_on_event(w * 0.97, cells), cells)
            self.assertTrue(circuit.fires_on_pulse(a * 1.03, cells), cells)
            self.assertFalse(circuit.fires_on_pulse(a * 0.97, cells), cells)

    def test_lesioned_cells_never_fire(self):
        for cells in circuit.CELLS:
            s = one(cells)
            s.alive[0] = False
            b = np.array([200.0])
            self.assertEqual(sum(len(s.step(b)) for _ in range(400)), 0, cells)

    def test_external_conductance_fires_a_cell(self):
        for cells in circuit.CELLS:
            s = one(cells, "FE")
            for _ in range(80):
                s.step()
            fired = len(s.step(None, np.array([0.5]), None))
            fired += sum(len(s.step()) for _ in range(80))
            self.assertGreaterEqual(fired, 1, cells)

    def test_pong_runs_on_every_cell(self):
        for cells in circuit.CELLS:
            r = pong.play(40, seed=0, cells=cells)
            self.assertEqual(r["neurons"], 107)


class Exchange(unittest.TestCase):
    def test_gamma(self):
        rng = np.random.default_rng(0)
        t = rng.random(20000) < 0.01
        self.assertAlmostEqual(float(exchange.gamma(t, t[None])[0]), 1.0)
        self.assertAlmostEqual(float(exchange.gamma(t, np.roll(t, 4)[None])[0]), 1.0)   # 1 ms off: inside 2 ms
        self.assertEqual(float(exchange.gamma(t, np.ones((1, t.size), bool))[0]), -1.0)
        r = rng.random((20, t.size)) < 0.01
        self.assertLess(abs(float(exchange.gamma(t, r).mean())), 0.05)
        self.assertEqual(float(exchange.gamma(t, np.zeros((1, t.size), bool))[0]), 0.0)

    def test_one_lif_fires_at_the_textbook_rate(self):
        sp = exchange.space("lifx1")
        p = dict(zip(sp.names, [1.0, 5.0, 10.0, 0.0, 2.0, 0.0, 0.0]))
        P = np.array([[p[k] for k in sp.names]])
        I = np.full(int(1000 / exchange.DT), 20.0)
        n = int(exchange.simulate("lifx1", P, I)[0].sum())
        period = 2.0 + 5.0 * np.log(20.0 / (20.0 - 10.0))      # refractory + time to threshold
        self.assertAlmostEqual(n, 1000 / period, delta=0.03 * 1000 / period)

    def test_output_delay_shifts_spikes(self):
        sp = exchange.space("lifx1")
        base = dict(zip(sp.names, [1.0, 5.0, 10.0, 0.0, 2.0, 0.0, 0.0]))
        I = np.full(400, 20.0)
        rows = []
        for lag in (0.0, 1.0):
            base["lag"] = lag
            rows.append(np.array([base[k] for k in sp.names]))
        a, b = exchange.simulate("lifx1", np.array(rows), I)
        self.assertTrue(np.array_equal(np.flatnonzero(a) + 4, np.flatnonzero(b)[: a.sum()]))

    def test_state_variables(self):
        self.assertEqual(exchange.state_variables("lifx1"), 1)
        self.assertEqual(exchange.state_variables("lifx4"), 7)
        self.assertEqual(exchange.state_variables("izh"), 2)


class Timing(unittest.TestCase):
    def test_counts_carry_no_class(self):
        tr, y = train.trials(400, 1)
        self.assertEqual(np.bincount(y).tolist(), [100] * 4)
        per = np.array([[sum(1 for f, _ in t if f == k) for k in range(train.FIBRES)] for t in tr])
        means = np.array([per[y == c].mean(0) for c in range(4)])
        self.assertLess(np.ptp(means, 0).max(), 0.15)           # the same counts in every class
        self.assertTrue((per >= 1).all())

    def test_silent_network_counts_nothing(self):
        tr, _ = train.trials(4, 1)
        W = np.zeros((2, train.FIBRES, 3))
        D = np.ones((2, train.FIBRES, 3))
        for cells in circuit.CELLS:
            self.assertEqual(train.counts(cells, W, D, tr).sum(), 0)

    def test_readout_separates_separable_counts(self):
        y = np.arange(40) % 4
        X = np.eye(4)[y][None] * 3.0
        self.assertTrue((train.readout(X, y, X)[0] == y).all())

    def test_training_beats_chance(self):
        r = train.train("lif", 2, seed=0, pop=8, elite=2, gens=3)
        self.assertGreater(r["test"], 0.35)


if __name__ == "__main__":
    unittest.main()
