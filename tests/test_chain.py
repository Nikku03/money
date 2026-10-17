import numpy as np

from brain import chain
from tests.util import TempHome

LINKS = [{"cause": "a", "effect": "b", "sign": 1, "weight": 1.0, "status": "untested"},
         {"cause": "b", "effect": "c", "sign": -1, "weight": 1.0, "status": "untested"},
         {"cause": "a", "effect": "d", "sign": 1, "weight": 0.5, "status": "tested"}]


class ChainTest(TempHome):
    def test_paths_table_and_encoding(self):
        ps = chain.paths(LINKS, "a", 4)
        self.assertEqual([[ln["effect"] for ln in p] for p in ps], [["b", "c"], ["d"]])  # strongest first
        nodes, succ = chain.table(ps[0], ["a", "b", "c", "d"])
        self.assertEqual(nodes[:4], ["a", "b", "c", "d"])
        self.assertEqual(list(succ), [1, 2, 2, 3, 4, 5])  # the chain's end and the fillers point to themselves
        X = chain.encode(succ, 0, 2)
        self.assertEqual(X.shape, (1, 2, chain.M * chain.M + chain.M))
        self.assertEqual(X[0, 0, chain.M * chain.M], 1.0)
        self.assertEqual(X[0, 1, chain.M * chain.M:].sum(), 0.0)  # the start is shown at step 0 only

    def test_train_save_load_and_follow(self):
        net = chain.train(steps=40, log=None)
        loaded = chain.load(chain.save(net))
        X = chain.encode(np.array([1, 2, 2, 3, 4, 5]), 0, 2)
        self.assertTrue(np.allclose(net.run(X, None, 0), loaded.run(X, None, 0)))
        rows = chain.follow(loaded, LINKS, "a", 4)
        self.assertEqual([r["exact"] for r in rows], [["b", "c"], ["d"]])
        self.assertEqual([r["sign"] for r in rows], [-1, 1])
        self.assertEqual([r["status"] for r in rows], ["untested", "tested"])
        self.assertTrue(all(len(r["engine"]) == len(r["exact"]) for r in rows))
