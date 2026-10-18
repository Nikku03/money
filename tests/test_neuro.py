import numpy as np

from brain import check
from brain.neuro import claims, memory, probe
from brain.neuro import subjects as S
from tests.util import TempHome

SMALL = {**S.MODADD, "p": 11, "hidden": 120, "steps": 30}


class ProbeTest(TempHome):
    def test_spectrum_reads_a_planted_cosine(self):
        p = 11
        x = np.arange(p)
        rows = np.stack([2.0 * np.cos(2 * np.pi * 3 * x / p + 0.7), np.cos(2 * np.pi * 5 * x / p - 1.2) + 0.5])
        sp = probe.spectrum(rows)
        self.assertEqual(sp["freq"].tolist(), [3, 5])
        self.assertTrue(np.allclose(sp["share"], 1.0))
        self.assertTrue(np.allclose(sp["phase"], [0.7, -1.2]))
        self.assertTrue(np.allclose(sp["amp"], [2.0, 1.0]))   # the constant 0.5 is not power

    def test_planted_network_has_the_mechanism_and_random_one_does_not(self):
        planted = S.planted_modadd(1, p=11, hidden=120)
        self.assertEqual(planted.accuracy(), 1.0)
        sp = probe.neurons(planted)
        self.assertGreater(probe.law_strength(sp, "sum", probe.same_freq(sp)), 0.99)
        self.assertLess(probe.law_strength(sp, "diff", probe.same_freq(sp)), 0.5)
        self.assertEqual(probe.sparse_stat(planted), 1.0)
        rand = S.train_modadd(1, {**SMALL, "steps": 0}, "untrained")
        self.assertLess(probe.sparse_stat(rand), 0.8)
        self.assertLess(np.mean(probe.null_sparse(planted, 20, 0)), 0.8)

    def test_projection_onto_the_planted_frequencies_changes_nothing(self):
        net = S.planted_modadd(2, p=11, hidden=120)
        fq = probe.neurons(net)["u"]["freq"]
        pj = probe.project(net, fq)
        self.assertTrue(np.allclose(pj.U, net.U) and np.allclose(pj.W, net.W))
        self.assertLess(max(probe.null_project(net, fq, 10, 0)), 0.5)

    def test_chain_probes_tell_the_two_planted_mechanisms_apart(self):
        look, mem = S.planted_lookup(), S.planted_memory()
        self.assertEqual(look.accuracy(16, n=200), 1.0)
        self.assertEqual(mem.accuracy(16, n=200), 1.0)
        self.assertEqual(probe.ablate(look, "h", n=300), 1.0)
        self.assertLess(probe.ablate(look, "e", n=300), 0.5)
        self.assertLess(probe.ablate(mem, "h", n=300), 0.5)
        self.assertEqual(probe.ablate(mem, "e", n=300), 1.0)
        self.assertEqual(probe.patch(look, n=300)["carries"], 1.0)
        self.assertEqual(probe.patch(mem, n=300)["stays"], 1.0)

    def test_the_engine_network_runs_through_the_subject_loop(self):
        s = S.train_chain(3, steps=5)
        from brain import chain
        X, _ = S.questions(np.random.default_rng(0), 50, 4)
        net = chain.load(chain.save(chain.train(5, seed=3, log=None)), seed=3)
        net.run(X, None, 0)
        self.assertTrue(np.array_equal(s.run(X), np.stack(net.items, 1)))
        nm = S.train_chain(3, memory=False, steps=5)
        self.assertEqual(nm.kind, "nomemory")
        X1 = X.copy()
        self.assertTrue(np.array_equal(nm.run(X1, hook_h=lambda t, h: h + 5.0), nm.run(X)))  # memory never read


class MemoryTest(TempHome):
    def test_rebuild_from_facts_and_transfer_to_a_new_modulus(self):
        bank = memory.neuro_bank()
        net = S.planted_modadd(3, p=11, hidden=120)
        ent = memory.record_modadd(bank, net)
        built = memory.rebuild(bank, ent)
        self.assertEqual(built.accuracy(), 1.0)
        self.assertTrue(np.allclose(built.W, net.W, atol=1e-3))   # W came from the law, not from W
        self.assertLess(memory.rebuild(bank, ent, law="diff").accuracy(), 0.5)
        self.assertGreaterEqual(memory.transfer(bank, ent, 13, 120, 0).accuracy(), 0.99)
        self.assertIn("rebuildable", memory.card(bank, ent))

    def test_no_mechanism_no_rebuild(self):
        bank = memory.neuro_bank()
        ent = memory.record_modadd(bank, S.train_modadd(1, {**SMALL, "steps": 0}, "untrained"))
        with self.assertRaises(memory.NoMechanism):
            memory.rebuild(bank, ent)
        self.assertIn("none recorded", memory.card(bank, ent))

    def test_the_neuro_bank_passes_the_bank_invariants(self):
        bank = memory.neuro_bank()
        memory.record_modadd(bank, S.planted_modadd(4, p=11, hidden=120))
        memory.record_chain(bank, S.planted_lookup(), units=False)
        errors, _ = check.run(bank)
        self.assertEqual(errors, [])


class ClaimsTest(TempHome):
    def test_specs_are_frozen(self):
        claims.declare(trial=True, log=None)
        self.assertEqual(claims.declare(trial=True, log=None), [])
        old = claims.CLAIMS[0]["threshold"]
        try:
            claims.CLAIMS[0]["threshold"] = 0.5
            with self.assertRaises(claims.Refused):
                claims.declare(trial=True, log=None)
        finally:
            claims.CLAIMS[0]["threshold"] = old

    def test_every_claim_has_a_measure_and_controls(self):
        ids = {c["id"] for c in claims.CLAIMS}
        self.assertEqual(len(ids), len(claims.CLAIMS))
        for c in claims.CLAIMS:
            self.assertIn(c["subject"], claims.LEARNT)
            self.assertTrue(c["positive"] and c["negative"])

    def test_a_planted_network_passes_the_modadd_claims(self):
        bank = memory.neuro_bank()
        net = S.planted_modadd(5)
        for cid in ("one-frequency", "same-frequency", "phase-sum", "faithful", "rebuild", "transfer"):
            self.assertTrue(claims.measure(cid, net, bank, 0)["pass"], cid)
