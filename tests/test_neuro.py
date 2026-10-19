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


class LanguageModelProbeTest(TempHome):
    """The parts of brain.neuro.lm that need no model (torch is optional)."""

    def test_planted_helix_is_fitted_read_and_beats_other_periods(self):
        from brain.neuro import lm, lmclaims
        P = lm.planted_states(256, seed=1)
        self.assertGreater(lm.helix_fit(P), 0.2)
        self.assertGreaterEqual(lm.helix_read(P), 0.9)
        self.assertLess(lm.helix_fit(P, labels=np.random.default_rng(0).permutation(100)), 0.0)
        self.assertLess(max(lm.helix_read(P, T) for T in lm.random_periods(5, 0)), 0.5)
        self.assertTrue(lmclaims.judge(lmclaims.CLAIMS[0], {
            "model": "x", "layer": 3, "counting_acc": 1.0, "helix-fit": {"stat": 0.3, "p_shuffle": 0.01, "p_random": 0.01},
            "planted": lmclaims.planted_control(64), "untrained": {"helix-fit": -0.1, "helix-read": 0.0}})["controls"]["pass"])

    def test_noise_has_no_helix(self):
        from brain.neuro import lm
        noise = np.random.default_rng(2).normal(size=(100, 256))
        self.assertLess(lm.helix_fit(noise), 0.0)
        self.assertLess(lm.helix_read(noise), 0.1)

    def test_basis_and_pairing(self):
        from brain.neuro import lm
        B = lm.basis([0, 1, 2])
        self.assertEqual(B.shape, (3, 8))       # n, cos for period 2, cos and sin for 5, 10 and 100
        self.assertTrue(np.allclose(B[:, 1], [1, -1, 1]))
        t = lm.targets()
        self.assertEqual(sorted(t), list(range(100)))
        self.assertTrue(all(t != np.arange(100)))


class LanguageProbeTest(TempHome):
    """brain.neuro.lang's measures and helpers, on made-up states (no model)."""

    def test_detect_by_script_and_common_words(self):
        from brain.neuro import lang as G
        self.assertEqual(G.detect("Это было очень давно"), "rus_Cyrl")
        self.assertEqual(G.detect("这是一个很好的问题"), "zho_Hans")
        self.assertEqual(G.detect("これはペンです"), "jpn_Jpan")
        self.assertEqual(G.detect("C'est une question que les gens posent dans la rue"), "fra_Latn")
        self.assertEqual(G.detect("Der Hund ist nicht in dem Haus und die Katze auch"), "deu_Latn")
        self.assertEqual(G.detect("The dog was in the house and it was warm"), "eng_Latn")
        self.assertEqual(G.detect(""), "unknown")
        self.assertEqual(G.detect("Zxq vrp"), "unknown")

    def test_first_sentence(self):
        from brain.neuro import lang as G
        self.assertEqual(G.first_sentence("One two. Three four."), "One two.")
        self.assertEqual(G.first_sentence("确保放松。然后继续。"), "确保放松。")
        self.assertEqual(G.first_sentence("no end"), "no end")

    def test_read_shared_and_random_like_on_planted_states(self):
        from brain.neuro import lang as G
        rng = np.random.default_rng(0)
        K, n, d = 4, 30, 64
        lang_dir = rng.normal(0, 3, (K, d))
        meaning = rng.normal(0, 1, (n, d))
        S = np.concatenate([meaning + lang_dir[k] + rng.normal(0, 0.1, (n, d)) for k in range(K)])   # language-major
        y = np.repeat(np.arange(K), n)
        self.assertEqual(G.read(S, y, S, y, K), 1.0)
        mu = G.means(S, y, K)
        self.assertGreater(G.shared(S, mu, n, K), 0.95)          # meaning matches once the language is removed
        shuffled = np.concatenate([S[k * n:(k + 1) * n][rng.permutation(n)] if k else S[:n] for k in range(K)])
        self.assertLess(G.shared(shuffled, mu, n, K), 0.3)
        r = G.random_like(lang_dir, 1)
        self.assertTrue(np.allclose(np.linalg.norm(r, axis=1), np.linalg.norm(lang_dir, axis=1)))
