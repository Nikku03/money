"""A billion-parameter language model as a subject: how does it hold a number while it counts? (docs/NEURO.md)

The model is shown ``"{c-2}, {c-1}, {c},"`` for c = 2..101 and continues with c + 1. Numbers up to 530 are single
tokens in the Pythia tokenizer, so the model cannot copy digits: it must know which number follows the token c.
Everything here reads or edits the residual stream at the token c, at one layer.

The known answer to grade against: language models hold a number n as a helix, its value plus waves of period
2, 5, 10 and 100 (Zhou et al. 2024, arXiv:2406.03445; Kantamneni & Tegmark 2025, arXiv:2502.00873, whose subjects
included Pythia-6.9B). The helix basis is ``B(n) = [n/100, cos and sin of 2 pi n / T for T in 5, 10, 100, cos(pi n)]``
(sin(pi n) is zero), nine numbers a number.

- ``helix_fit``: how well the basis predicts the states of numbers held out of the fit (10-fold R^2).
- ``helix_read``: fit on every other number, project a held-out number's state onto the fitted directions and
  decode the nearest helix point. Is the number in that subspace exactly?
- ``patch``: replace the state at c with the state another prompt had at its number m. Does the model count on
  from m?
- ``steer``: keep the prompt's own state but move only its helix part from n to m (the fit never saw m). How much
  of the logit gap toward m + 1 does that recover, and does the model then write m + 1?

Weights come from Hugging Face (EleutherAI, Apache-2.0), pinned to the commit the declaration names, cached under
``<home>/hf``. ``step0`` is each model's published untrained checkpoint: same architecture and tokenizer, no
learning, the negative control. Needs torch and transformers (``pip install -e .[lm]``).
"""
from __future__ import annotations

import os

import numpy as np

from brain import config
from brain.bank import Bank, Fact
from brain.timeutil import now_iso

PERIODS = (2, 5, 10, 100)
POOL = (3, 4, 6, 7, 8, 9, 11, 12, 13, 15, 20, 25, 50)   # periods the random-period null draws 4 of
NUMS = np.arange(2, 102)
PAIR = 37      # prompt k is patched or steered toward number k + 37 (mod 100)
PAIR_LAYER = 61  # a different pairing, used only to choose the layer


def _hf_env():
    os.environ.setdefault("HF_HOME", str(config.home() / "hf"))


def basis(n, periods=PERIODS) -> np.ndarray:
    n = np.asarray(n, float)
    cols = [n / 100.0]
    for t in periods:
        cols.append(np.cos(2 * np.pi * n / t))
        if t != 2:
            cols.append(np.sin(2 * np.pi * n / t))
    return np.stack(cols, 1)


def pinned_sha(name: str, revision: str = "main") -> str:
    _hf_env()
    from huggingface_hub import model_info
    return model_info(name, revision=revision).sha


class LM:
    """One Pythia model, its counting prompts, and the states at the number token at every layer."""

    def __init__(self, name: str, revision: str = "main", sha: str | None = None):
        _hf_env()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import logging
        logging.set_verbosity_error()
        logging.disable_progress_bar()
        self.torch = torch
        self.name, self.revision = name, revision
        self.sha = sha or pinned_sha(name, revision)
        self.tok = AutoTokenizer.from_pretrained(name, revision=self.sha)
        self.model = AutoModelForCausalLM.from_pretrained(name, revision=self.sha, dtype=torch.float32).eval()
        self.tok.padding_side = "left"
        self.tok.pad_token = self.tok.eos_token
        self.layers = self.model.gpt_neox.layers
        self.n_layers = self.model.config.num_hidden_layers
        self.params = int(sum(p.numel() for p in self.model.parameters()))
        self.enc = self.tok([f"{c - 2}, {c - 1}, {c}," for c in NUMS], return_tensors="pt", padding=True)
        self.ans = np.array([self.tok.encode(f" {c + 1}")[0] for c in NUMS])
        with torch.no_grad():
            out = self.model(**self.enc, output_hidden_states=True)
        self.base = out.logits[:, -1].numpy().astype(np.float64)
        # H[i]: the state at the number token going into layer i (H[0] is the embedding)
        self.H = [h[:, -2].numpy().astype(np.float64) for h in out.hidden_states]

    @property
    def entity(self) -> str:
        return f"LM-{self.name.split('/')[-1]}-{self.revision}".upper()

    def accuracy(self) -> float:
        return float((self.base.argmax(1) == self.ans).mean())

    def logits_with(self, layer: int, states: np.ndarray) -> np.ndarray:
        """Final-position logits with the state at the number token, going into ``layer``, replaced."""
        torch = self.torch

        def hook(mod, args, kwargs):
            hs = (args[0] if args else kwargs["hidden_states"]).clone()
            hs[:, -2] = torch.tensor(states, dtype=hs.dtype)
            if args:
                return (hs,) + args[1:], kwargs
            kwargs["hidden_states"] = hs
            return args, kwargs
        h = self.layers[layer].register_forward_pre_hook(hook, with_kwargs=True)
        try:
            with torch.no_grad():
                return self.model(**self.enc).logits[:, -1].numpy().astype(np.float64)
        finally:
            h.remove()


# ---------- looking ----------
def _fit(B, H, idx):
    C, *_ = np.linalg.lstsq(np.c_[np.ones(len(idx)), B[idx]], H[idx], rcond=None)
    return C[0], C[1:]


def helix_fit(H, periods=PERIODS, folds: int = 10, seed: int = 0, labels=None) -> float:
    """Held-out R^2 of the basis predicting the states (``labels`` permutes which number each state belongs to)."""
    n = NUMS if labels is None else NUMS[labels]
    B = basis(n, periods)
    perm = np.random.default_rng(seed).permutation(len(H))
    pred = np.zeros_like(H)
    for f in range(folds):
        te = perm[f::folds]
        mu, C = _fit(B, H, np.setdiff1d(perm, te))
        pred[te] = mu + B[te] @ C
    return float(1 - ((H - pred) ** 2).sum() / ((H - H.mean(0)) ** 2).sum())


def helix_read(H, periods=PERIODS) -> float:
    """Share of numbers decoded exactly from their own state by a fit that never saw them."""
    B = basis(NUMS, periods)
    ok = 0
    for k in range(len(H)):
        mu, C = _fit(B, H, np.setdiff1d(np.arange(len(H)), [k]))
        f, *_ = np.linalg.lstsq(C.T, H[k] - mu, rcond=None)
        ok += int(np.argmin(((B - f) ** 2).sum(1)) == k)
    return ok / len(H)


def random_periods(n: int, seed: int) -> list[tuple]:
    rng = np.random.default_rng(seed)
    return [tuple(int(x) for x in rng.choice(POOL, 4, replace=False)) for _ in range(n)]


def planted_states(d: int, seed: int, helix_share: float = 0.4) -> np.ndarray:
    """A positive control: states that are a helix in random directions plus noise, the helix about
    ``helix_share`` of the variance."""
    rng = np.random.default_rng(seed)
    B = basis(NUMS)
    S = (B - B.mean(0)) @ rng.normal(0, 1, (B.shape[1], d))
    noise_sd = np.sqrt(S.var(0).sum() * (1 - helix_share) / helix_share / d)
    return S + rng.normal(0, noise_sd, (len(NUMS), d))


# ---------- intervening ----------
def targets(pair: int = PAIR) -> np.ndarray:
    return (np.arange(len(NUMS)) + pair) % len(NUMS)


def patch(lm: LM, layer: int, pair: int = PAIR) -> float:
    """Share of prompts that count on from m when the state at their number is the state m had."""
    t = targets(pair)
    got = lm.logits_with(layer, lm.H[layer][t]).argmax(1)
    return float((got == lm.ans[t]).mean())


def choose_layer(lm: LM) -> tuple[int, list[float]]:
    """The layer where patching the real state works best, on the layer-choice pairing; the earliest on a tie."""
    accs = [patch(lm, i, PAIR_LAYER) for i in range(1, lm.n_layers)]
    return 1 + int(np.argmax(accs)), accs


def steer(lm: LM, layer: int, periods=PERIODS, labels=None) -> dict:
    """Move only the helix part of each prompt's state from its number n to m = target (fit without m).
    ``restored``: median share of the logit gap (m+1 over n+1) that the real state of m opens;
    ``writes``: share of prompts whose top token is then m+1; ``stays``: still n+1."""
    H = lm.H[layer]
    n_of = NUMS if labels is None else NUMS[labels]
    B = basis(n_of, periods)
    t = targets()
    S = np.zeros_like(H)
    for k in range(len(H)):
        _, C = _fit(B, H, np.setdiff1d(np.arange(len(H)), [t[k]]))
        S[k] = H[k] + (B[t[k]] - B[k]) @ C
    lo = lm.logits_with(layer, S)
    real = lm.logits_with(layer, H[t])
    r = np.arange(len(H))

    def gap(x):
        return x[r, lm.ans[t]] - x[r, lm.ans]
    g0 = gap(lm.base)
    restored = (gap(lo) - g0) / np.where(np.abs(gap(real) - g0) > 1e-9, gap(real) - g0, np.nan)
    top = lo.argmax(1)
    return {"restored": float(np.nanmedian(restored)), "writes": float((top == lm.ans[t]).mean()),
            "stays": float((top == lm.ans).mean())}


def amplitudes(H) -> dict:
    """How much of the number's state each period carries: the norm of its fitted directions."""
    B = basis(NUMS)
    _, C = _fit(B, H, np.arange(len(H)))
    out, j = {"linear": float(np.linalg.norm(C[0]))}, 1
    for t in PERIODS:
        k = 1 if t == 2 else 2
        out[f"T{t}"] = float(np.linalg.norm(C[j:j + k]))
        j += k
    return out


# ---------- the bank ----------
def bank() -> Bank:
    return Bank(config.home() / "neuro" / "bank")


def record(b: Bank, lm: LM, findings: dict) -> str:
    """The model as an entity (its subject fact cites the pinned Hugging Face commit) and each finding as a
    derived fact citing it."""
    t = now_iso()
    subj = Fact(lm.entity, "subject", {"task": "counting", "model": lm.name, "revision": lm.revision,
                                       "sha": lm.sha, "params": lm.params, "layers": lm.n_layers,
                                       "counting_acc": round(lm.accuracy(), 4)},
                t[:10], t, "reported", {"name": "huggingface", "url": f"https://huggingface.co/{lm.name}/tree/{lm.sha}"})
    facts = [subj] + [Fact(lm.entity, k, v, t[:10], t, "derived", {"name": f"brain.neuro.lm:{k.split('/')[0]}"},
                           inputs=(subj.ref,)) for k, v in findings.items()]
    b.add_many(facts)
    return lm.entity


def card(b: Bank, ent: str) -> str:
    fs: dict[str, Fact] = {}
    for f in b.facts(ent):
        if f.attr not in fs or f.known_at >= fs[f.attr].known_at:
            fs[f.attr] = f
    if "subject" not in fs:
        return f"{ent}: not in the neuro bank"
    s = fs["subject"]
    v = s.value
    lines = [(f"{ent}  {v['model']} @ {v['sha'][:10]}  {v['params'] / 1e9:.2f}B parameters, {v['layers']} layers  "
              f"[{s.id}]"), f"  counting accuracy {v['counting_acc']:.3f}"]
    for attr, fmt in (("layer/chosen", lambda x: f"number read at layer {x['layer']} (real-state patch {x['patch']:.2f})"),
                      ("helix/fit", lambda x: f"helix predicts held-out numbers' states: R^2 {x['r2']:.3f}"),
                      ("helix/read", lambda x: f"number read back exactly from the helix part: {x['acc']:.2f}"),
                      ("helix/steer", lambda x: f"helix shift recovers {x['restored']:.2f} of the gap; writes m+1 "
                                                f"{x['writes']:.2f}, keeps n+1 {x['stays']:.2f}"),
                      ("helix/amplitudes", lambda x: "fitted size per part: " + ", ".join(f"{k} {val:.1f}" for k, val in x.items()))):
        if attr in fs:
            lines.append(f"  {fmt(fs[attr].value)}  [{fs[attr].id}]")
    return "\n".join(lines)
