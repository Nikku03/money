"""The hand-wired spiking circuit (brain/neurons): cells, delays, simplices, the Pong wiring."""
import unittest

import numpy as np

from brain.neurons import pong
from brain.neurons.circuit import Circuit
from brain.neurons.topology import max_dimension, simplices


def kick(c, sim, srcs, ms=40):
    """Fire ``srcs`` once at t=0; return {neuron: [spike times]}."""
    out = {}
    for k in range(int(ms / c.dt)):
        b = np.zeros(len(c.neurons))
        if k == 0:
            b[list(srcs)] = 2000.0
        for n in sim.step(b):
            out.setdefault(int(n), []).append(k * c.dt)
    return out


class CellTest(unittest.TestCase):
    def test_rest_and_tonic_firing(self):
        c = Circuit()
        c.add("a")
        sim = c.compile()
        self.assertEqual(sum(len(sim.step(np.array([0.0]))) for _ in range(2000)), 0)
        self.assertGreater(sum(len(sim.step(np.array([10.0]))) for _ in range(4000)), 5)

    def test_delay_sets_arrival(self):
        for delay in (2.0, 8.0):
            c = Circuit()
            a, b = c.add("a"), c.add("b", "FE")
            c.connect(a, b, 0.5, delay)
            t = kick(c, c.compile(), [a])
            self.assertGreater(t[b][0], delay)          # cannot fire before the spike arrives
            self.assertLess(t[b][0], delay + 6)

    def test_coincidence(self):
        c = Circuit()
        a, b, x = c.add("a"), c.add("b"), c.add("x", "FE")
        c.connect(a, x, pong.EYE_W, 1.0)
        c.connect(b, x, pong.EYE_W, 1.0)
        self.assertNotIn(x, kick(c, c.compile(), [a]))
        self.assertIn(x, kick(c, c.compile(), [a, b]))

    def test_inhibition_vetoes(self):
        c = Circuit()
        a, i, x = c.add("a"), c.add("i", "FS", sign=-1), c.add("x", "FE")
        c.connect(a, x, 0.2, 2.0)
        c.connect(i, x, 1.0, 1.0)
        self.assertIn(x, kick(c, c.compile(), [a]))
        self.assertNotIn(x, kick(c, c.compile(), [a, i]))

    def test_rules(self):
        c = Circuit()
        a = c.add("a")
        with self.assertRaises(ValueError):
            c.connect(a, a, 0.1)
        b = c.add("b")
        with self.assertRaises(ValueError):
            c.connect(a, b, -0.1)
        with self.assertRaises(ValueError):
            c.connect(a, b, 0.1, delay=0.0)


class TopologyTest(unittest.TestCase):
    def test_simplex_counts(self):
        adj = {i: set(range(i + 1, 12)) for i in range(12)}
        cnt = simplices(adj)
        self.assertEqual(cnt[11], 1)
        self.assertEqual(cnt[1], 66)
        self.assertEqual(cnt[2], 220)
        self.assertEqual(max_dimension({0: {1}, 1: {2}, 2: {0}}), 1)   # a cycle has no triangle

    def test_pong_circuit_reaches_dimension_11(self):
        c = pong.build()
        self.assertEqual(simplices(c.adjacency())[11], 2)                 # UP and DOWN
        up = c.group("up")
        self.assertEqual({s.pre for s in c.inputs(up[11])}, set(up[:11]))  # the sink hears all eleven


class PongTest(unittest.TestCase):
    def test_clique_volley_reaches_sink(self):
        c = pong.build()
        up = c.group("up")
        t = kick(c, c.compile(), [up[0]], ms=60)
        self.assertTrue(all(n in t for n in up))
        times = [t[n][0] for n in up]
        self.assertEqual(times, sorted(times))                           # in order, source to sink

    def test_direction(self):
        c = pong.build()
        sim = c.compile()
        drive = np.zeros(len(c.neurons))
        drive[c.group("ball")] = pong.place_drive(0.85)
        drive[c.group("paddle")] = pong.place_drive(0.2)
        up = down = 0
        for f in range(5):
            for k in range(int(pong.FRAME_MS / c.dt)):
                fired = sim.step(drive if k < 4 else None)
                up += c.group("up")[-1] in fired
                down += c.group("down")[-1] in fired
        self.assertGreater(up, 0)
        self.assertEqual(down, 0)

    def test_plays_better_than_standing_still(self):
        n = pong.play(400, seed=0)
        s = pong.play(400, seed=0, controller="still")
        self.assertGreater(n["hits"], 0)
        self.assertEqual(n["misses"], 0)
        self.assertLess(s["hits"] / max(1, s["hits"] + s["misses"]), 1.0)

    def test_replay(self):
        r = pong.play(5, record=True)
        self.assertEqual(len(r["replay"]), 5)
        self.assertEqual(len(r["traced"]), len(r["replay"][0]["v"][0]))


if __name__ == "__main__":
    unittest.main()


class CatchTest(unittest.TestCase):
    def test_fold_and_landing(self):
        from brain.neurons import catch
        self.assertAlmostEqual(catch.fold(1.2), 0.8)
        self.assertAlmostEqual(catch.fold(-0.3), 0.3)
        self.assertAlmostEqual(catch.fold(2.5), 0.5)
        # Served from the middle, rising 1/30 per frame: up 1.0 in 30 frames, one bounce, lands at 0.5.
        self.assertAlmostEqual(catch.landing(0.0, 0.5, 1 / 30), 0.5)
        self.assertAlmostEqual(catch.landing(0.0, 0.5, 1 / 30, walls=False), 1.0)

    def test_game_lands_where_predicted(self):
        from brain.neurons import catch
        g = catch.Game(seed=4)
        want = g.landing()
        while g.bx + catch.VX < 1.0 - 1e-9:
            g.tick()
        self.assertAlmostEqual(g.by + g.vy if 0 <= g.by + g.vy <= 1 else catch.fold(g.by + g.vy), want, places=6)

    def test_motion_cell_needs_the_delayed_spike(self):
        from brain.neurons import catch
        c = catch.build()
        r = c.by_name("R9_+2")
        ins = {c.neurons[s.pre].name: s.delay for s in c.inputs(r)}
        self.assertEqual(ins, {"Y9": 1.0, "Y7": catch.K * catch.FRAME_MS + 1.0})
        # Y7 then, Y9 six frames later: the R cell for +2 fires; Y9 alone does not fire it.
        sim = c.compile()
        hits = 0
        for f in range(catch.K + 1):
            for k in range(int(catch.FRAME_MS / c.dt)):
                b = np.zeros(len(c.neurons))
                if k < 4:
                    b[c.by_name("Y7" if f == 0 else "Y9" if f == catch.K else "Y0")] = pong.PULSE_AMP
                hits += r in sim.step(b)
        self.assertEqual(hits, 1)

    def test_predict_cells_feed_their_landing(self):
        from brain.neurons import catch
        c = catch.build()
        outs = [c.neurons[s.post].name for s in c.outputs(c.by_name("P3_9_+2"))]
        self.assertEqual(outs, ["L7"])            # 0.59 + 0.0208 * 16.9 frames = 0.946 -> band 7

    def test_prediction_beats_chasing(self):
        from brain.neurons import catch
        n = catch.play(300, seed=10)
        r = catch.play(300, seed=10, controller="reactive")
        self.assertGreater(n["hits"], r["hits"])
        self.assertLess(n["guess_error"]["near"], catch.PADDLE_H / 2)
