"""How a large language model tells languages apart, and whether that can be read and written (docs/NEURO.md).

The data is FLORES passages as Belebele publishes them (facebook/belebele, CC-BY-SA 4.0): the same sentence in
twelve languages, so meaning is held fixed while the language changes. Each passage's first sentence is shown as
``"Text: {sentence}\\nLanguage:"`` and the model names the language (`` English``, `` French``, ...; each name is
one token in the Qwen tokenizer).

Probes, each at one layer of the residual stream:
- ``read``: is the language in the state at the final ``:`` (a token identical in every prompt, so anything there
  the model put there)? A nearest-mean classifier fitted on some passages, scored on others.
- ``swap``: replace that state's language part (h - mean of its language + mean of another). Does the model then
  name the other language?
- ``shared``: take each sentence's state, averaged over its own tokens, minus its language's mean. Is what is
  left the same for a sentence and its translations (the meaning, without the language)?
- ``speak``: add the difference between another language's mean and English's to every position while the model
  continues an English sentence. Does it continue in the other language?

The known answer to grade against: in multilingual models, middle layers hold meaning largely independent of
language, with language as a separate, roughly additive part (Wendler et al. 2024, arXiv:2402.10588; Dumas et al.
2024, arXiv:2411.08745; language-specific neurons, Tang et al. 2024, arXiv:2402.16438).

Works on any Hugging Face causal model whose decoder layers are one ``nn.ModuleList`` (Llama, Qwen, Mistral,
Gemma). Ollama serves quantized GGUF builds and shows only text, so the probes load the same model's open
weights from Hugging Face instead.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

from brain import config

LANGS = [("eng_Latn", "English"), ("fra_Latn", "French"), ("spa_Latn", "Spanish"), ("deu_Latn", "German"),
         ("ita_Latn", "Italian"), ("por_Latn", "Portuguese"), ("nld_Latn", "Dutch"), ("rus_Cyrl", "Russian"),
         ("zho_Hans", "Chinese"), ("jpn_Jpan", "Japanese"), ("hin_Deva", "Hindi"), ("arb_Arab", "Arabic")]
SPEAK = ["fra_Latn", "deu_Latn", "spa_Latn", "rus_Cyrl", "zho_Hans"]   # targets for steering an English text
DATASET, DATASET_SHA = "facebook/belebele", "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"
PREFIX, SUFFIX = "Text: ", "\nLanguage:"
ALPHAS = (1.0, 2.0, 4.0, 8.0)   # steering strengths tried for speak, as multiples of the language difference
NEW_TOKENS = 20

_END = re.compile(r"(.+?[.!?。！？।؟])(\s|$)", re.S)

STOP = {   # a few of the commonest words, enough to tell these Latin-script languages apart in 20 tokens
    "eng_Latn": "the and of to is in that it was for on are with as".split(),
    "fra_Latn": "le la les et des est une un que du dans pour pas sur qui".split(),
    "spa_Latn": "el la los las y de que en es un una por con para del".split(),
    "deu_Latn": "der die das und ist nicht ein eine zu den mit sich auf des dem".split(),
    "ita_Latn": "il la di e che è un una per non del della con sono gli".split(),
    "por_Latn": "o a os as e de que em um uma para não do da com".split(),
    "nld_Latn": "de het een en van is dat niet op te zijn met voor".split(),
}


def _hf_env():
    os.environ.setdefault("HF_HOME", str(config.home() / "hf"))


def first_sentence(text: str) -> str:
    m = _END.match(text.strip())
    s = m.group(1) if m else text.strip()
    return s[:400]


def passages(n: int) -> dict[str, list[str]]:
    """The first ``n`` passages (in English order) that every language has, as their first sentence."""
    _hf_env()
    from huggingface_hub import hf_hub_download
    rows = {}
    for code, _ in LANGS:
        p = hf_hub_download(DATASET, f"data/{code}.jsonl", repo_type="dataset", revision=DATASET_SHA)
        by = {}
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            by.setdefault(r["link"], r["flores_passage"])
        rows[code] = by
    links = [k for k in rows["eng_Latn"] if all(k in rows[c] for c, _ in LANGS)][:n]
    return {code: [first_sentence(rows[code][k]) for k in links] for code, _ in LANGS}


def detect(text: str) -> str:
    """The language of a short generated text: by script, then by common words; ``unknown`` if nothing fits."""
    t = text.strip()
    if not t:
        return "unknown"
    count = lambda lo, hi: sum(lo <= ord(ch) <= hi for ch in t)   # noqa: E731
    scripts = {"rus_Cyrl": count(0x0400, 0x04FF), "hin_Deva": count(0x0900, 0x097F), "arb_Arab": count(0x0600, 0x06FF),
               "jpn_Jpan": count(0x3040, 0x30FF), "zho_Hans": count(0x4E00, 0x9FFF)}
    if scripts["jpn_Jpan"] >= 2:
        return "jpn_Jpan"
    best = max(scripts, key=scripts.get)
    if scripts[best] >= 3:
        return best
    words = re.findall(r"[a-zà-ÿ']+", t.lower())
    hits = {c: sum(w in set(sw) for w in words) for c, sw in STOP.items()}
    best = max(hits, key=hits.get)
    return best if hits[best] >= 2 and list(hits.values()).count(hits[best]) == 1 else "unknown"


class Model:
    """A causal LM with its decoder layers found, the language-name tokens, and hooks to edit one layer's input."""

    def __init__(self, name: str, sha: str, quant: str | None = None, device: str | None = None):
        _hf_env()
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import logging
        logging.set_verbosity_error()
        self.torch = torch
        self.name, self.sha, self.quant = name, sha, quant
        cuda = torch.cuda.is_available() if device is None else device == "cuda"
        kw = {"dtype": torch.bfloat16 if cuda else torch.float32}
        if quant == "nf4":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16)
        if cuda:
            kw["device_map"] = "auto"
        self.tok = AutoTokenizer.from_pretrained(name, revision=sha)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name, revision=sha, **kw).eval()
        cfg = self.model.config.get_text_config()
        self.n_layers = cfg.num_hidden_layers
        self.layers = next(m for m in self.model.modules()
                           if isinstance(m, torch.nn.ModuleList) and len(m) == self.n_layers)
        self.params = int(sum(p.numel() for p in self.model.parameters()))
        self.device = next(self.model.parameters()).device
        ids = [self.tok.encode(" " + nm, add_special_tokens=False)[0] for _, nm in LANGS]
        if len(set(ids)) != len(LANGS):
            raise ValueError(f"{name}: the language names do not start with distinct tokens")
        self.name_ids = np.array(ids)
        self.n_prefix = len(self.tok.encode(PREFIX.strip(), add_special_tokens=False))
        self.n_suffix = len(self.tok.encode(SUFFIX, add_special_tokens=False))

    def _enc(self, texts):
        return self.tok(texts, return_tensors="pt", padding=True).to(self.device)

    def states(self, sentences: list[str], batch: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For each sentence's prompt: the state at the last token at every layer (n, layers + 1, d), the state
        averaged over the sentence's own tokens (same shape), and the model's top token."""
        torch = self.torch
        last, pooled, top = [], [], []
        for i in range(0, len(sentences), batch):
            enc = self._enc([PREFIX + s + SUFFIX for s in sentences[i:i + batch]])
            with torch.no_grad():   # logits for the last position only: the vocabulary is ~150k wide
                out = self.model(**enc, output_hidden_states=True, logits_to_keep=1)
            m = enc["attention_mask"].clone()
            T = m.shape[1]
            lengths = m.sum(1)
            for r in range(m.shape[0]):   # keep only the sentence's tokens: drop padding, prefix and suffix
                start = T - int(lengths[r]) + self.n_prefix
                m[r, :start] = 0
                m[r, T - self.n_suffix:] = 0
            w = (m / m.sum(1, keepdim=True).clamp(min=1)).to(out.hidden_states[0].dtype)
            hs = torch.stack(out.hidden_states, 1)                     # (b, layers + 1, T, d)
            last.append(hs[:, :, -1].float().cpu().numpy())
            pooled.append(torch.einsum("bt,bltd->bld", w, hs).float().cpu().numpy())
            top.append(out.logits[:, -1].argmax(-1).cpu().numpy())
            del out, hs
        return np.concatenate(last), np.concatenate(pooled), np.concatenate(top)

    def _hook(self, layer: int, edit):
        torch = self.torch

        def pre(mod, args, kwargs):
            hs = args[0] if args else kwargs["hidden_states"]
            hs = edit(hs.clone())
            if args:
                return (hs,) + args[1:], kwargs
            kwargs["hidden_states"] = hs
            return args, kwargs
        del torch
        return self.layers[layer].register_forward_pre_hook(pre, with_kwargs=True)

    def top_with_last(self, sentences: list[str], layer: int, new_last: np.ndarray, batch: int = 16) -> np.ndarray:
        """The top token when the state at the last position, going into ``layer``, is replaced by ``new_last``."""
        torch = self.torch
        tops = []
        for i in range(0, len(sentences), batch):
            v = torch.tensor(new_last[i:i + batch], device=self.device)

            def edit(hs, v=v):
                hs[:, -1] = v.to(hs.dtype)
                return hs
            h = self._hook(layer, edit)
            try:
                with torch.no_grad():
                    tops.append(self.model(**self._enc([PREFIX + s + SUFFIX for s in sentences[i:i + batch]]),
                                           logits_to_keep=1).logits[:, -1].argmax(-1).cpu().numpy())
            finally:
                h.remove()
        return np.concatenate(tops)

    def continue_with(self, texts: list[str], layer: int | None = None, add: np.ndarray | None = None,
                      batch: int = 16) -> list[str]:
        """Greedy continuations, with ``add[i]`` added to every position going into ``layer`` (none if None)."""
        torch = self.torch
        outs = []
        for i in range(0, len(texts), batch):
            enc = self._enc(texts[i:i + batch])
            h = None
            if add is not None:
                v = torch.tensor(add[i:i + batch], device=self.device)

                def edit(hs, v=v):
                    return hs + v[:, None, :].to(hs.dtype)
                h = self._hook(layer, edit)
            try:
                with torch.no_grad():
                    g = self.model.generate(**enc, max_new_tokens=NEW_TOKENS, do_sample=False,
                                            pad_token_id=self.tok.pad_token_id)
            finally:
                if h is not None:
                    h.remove()
            outs += self.tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return outs


# ---------- the measures, on states already collected ----------
def means(S: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    return np.stack([S[labels == c].mean(0) for c in range(k)])


def read(S_train, y_train, S_test, y_test, k: int) -> float:
    """Nearest-mean accuracy: language means from one set of passages, scored on another."""
    mu = means(S_train, y_train, k)
    d = ((S_test[:, None, :] - mu[None]) ** 2).sum(-1)
    return float((d.argmin(1) == y_test).mean())


def shared(P_test: np.ndarray, mu: np.ndarray, n: int, k: int, remove: bool = True) -> float:
    """Translation retrieval: for each sentence in language a, is its nearest sentence in language b (by cosine,
    language means removed) its own translation? Averaged over every ordered pair of languages. ``P_test`` is
    (k * n, d), language-major; chance is 1/n."""
    X = P_test.reshape(k, n, -1) - (mu[:, None, :] if remove else 0)
    X = X / np.linalg.norm(X, axis=-1, keepdims=True).clip(min=1e-9)
    acc = []
    for a in range(k):
        for b in range(k):
            if a != b:
                acc.append(float(((X[a] @ X[b].T).argmax(1) == np.arange(n)).mean()))
    return float(np.mean(acc))


def random_like(delta: np.ndarray, seed: int) -> np.ndarray:
    """Random directions with the same norms as the rows of ``delta``."""
    r = np.random.default_rng(seed).normal(size=delta.shape)
    return r / np.linalg.norm(r, axis=-1, keepdims=True) * np.linalg.norm(delta, axis=-1, keepdims=True)
