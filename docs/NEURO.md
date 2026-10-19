# Looking inside a neural network

Can the memory bank and the evidence engine understand a neural network? Hand them one and see whether they find out how it works inside, and whether what they record is enough to build another one with the same properties.

```
python -m brain neuro look      # dev networks: probe them, write what is found to the neuro bank, print a card each
python -m brain neuro declare   # write every claim's spec, with the code's sha, to research/neuro_ledger.jsonl
git commit ...                  # the declaration must be in HEAD before it runs
python -m brain neuro test      # train fresh networks, test every claim, write the ledger and research/NEURO.md
python -m brain neuro card      # what the bank holds about each network, every line citing its fact
```

`--trial` runs `declare` and `test` on the dev networks with a scratch ledger, to exercise the code without touching the record. The results are in [`research/NEURO.md`](../research/NEURO.md).

## The same rules as for a market claim

A network is treated like a company: an entity in a bank, with facts about it that cite where they came from, and claims about it that count only once they have been tested.

| For a stock | For a network |
| --- | --- |
| The entity is a ticker | The entity is `NN-<sha of its weights>`; its `subject` fact cites the code that trained it |
| Facts from filings and prices | Facts from probes: each neuron's spectrum, the phase law, ablation and patching results; each one derived, citing the subject fact |
| TRAIN years for looking, TEST years for the verdict | Dev networks for looking (one per subject), ten networks from fresh seeds for the verdict |
| A shuffled signal as the null | A null that keeps everything but the structure in question (weights shuffled over inputs, output weights paired with other neurons, phases shuffled, the wrong frequency kept) |
| An oracle signal must pass, random ones pass about 5% of the time | A hand-built network with the mechanism must pass; a network without it must fail |
| Declared before running, frozen, false-discovery correction | Same; the claims keep their own family (`research/neuro_ledger.jsonl`), since near-zero p-values in the market family would loosen every market hypothesis's q |

Interventions are what separate an explanation from a description. A description says "each neuron looks like a cosine". The intervention cuts every neuron down to that cosine and checks the network still gives its own answers. If it does, the description is the computation, not just a pattern in the weights.

## The two networks

**Modular addition** (`brain/neuro/subjects.py`, `modadd`). A one-hidden-layer network computes (a + b) mod 23: inputs one-hot a and b, hidden `h = (U[:, a] + V[:, b])²`, logits `W h`. Trained on half the 529 pairs, with weight decay, it gets the other half right too. The literature has worked out how such networks do it (Nanda et al. 2023, [arXiv:2301.05217](https://arxiv.org/abs/2301.05217); Gromov 2023, [arXiv:2301.02679](https://arxiv.org/abs/2301.02679)):
- each neuron reads a and b through a cosine of one frequency;
- its output weights are a cosine of the same frequency;
- the output phase is the sum of the two input phases, so the product of the input cosines becomes cos(a + b).

That makes it a known answer to grade the engine against.
- The positive control is that construction written by hand (`planted`).
- The negative control is the same network trained with almost no weight decay (`memorizer`): it fits the training pairs and fails the rest.

**The chain engine's own network** (`brain.chain.train`, `chain`). The bump-node tree with the snap, copied from Nikku03/cell. Each step reads three things:
- the table (which item leads to which);
- the written-down pointer `e`, a learnt code for the item it is on;
- a continuous memory `h`.

It writes the next memory and item. It is exact on chains far longer than it was trained on, but how it follows a chain was never written down. The controls are two hand-written steps:
- one that looks the pointer's row up in the table and ignores the memory (`planted_lookup`);
- one that keeps the item in its memory and ignores the pointer (`planted_memory`).

They behave the same from outside, so only probes of the inside can tell them apart.

## What is recorded

Each probe writes facts to `<home>/neuro/bank` (the same `Bank` as the market bank, in a separate directory; `brain check` invariants hold there too):

- modular addition:
  - `neuron/017`: each neuron's top frequency, its share of the power, its phase and its amplitude, in U, V and W;
  - `law/phase`: how strongly "out = a + b" and "out = a − b" hold;
  - `spectrum/summary`: the medians and the number of neurons per frequency;
- the chain engine:
  - `ablate/none|memory|pointer`: 16-step accuracy with the memory or the pointer replaced by its mean at every step;
  - `patch/pointer`: with the pointer switched to item j's code, how often the network writes table[j] and carries on from there;
  - `unit/07`: 16-step accuracy without each output unit.

## Making it again from the facts

`memory.rebuild` writes a modular-addition network from one entity's facts alone:
- each neuron is a cosine of its recorded frequency, with the input phases and amplitudes as recorded (six numbers a neuron instead of 69 weights);
- the output weights come from the recorded phase law, never from the network's W.

It refuses (`NoMechanism`) when the facts do not describe single-cosine neurons with a phase law that holds.

`memory.transfer` goes further: it reads only the rule (one frequency per neuron, the law) and writes a network for modulus 29, which nothing was trained on. If that works, what was recorded is an algorithm, not a compressed copy.

For the chain engine, the replication is a prediction. If the probes are right that the memory carries nothing, the same tree without a memory should be just as exact. `chain-rebuild` trains it and checks.

## The claims

Ten claims, `N1-*`, in `brain/neuro/claims.py`. Each is tested on ten networks from fresh seeds (modular addition seeds 1–10, the chain engine seeds 30–39).
- **Per network:** the measure must clear its threshold, and must beat its null at p < 0.05 where the claim has one.
- **PASS** needs all of:
  - at least 9 of the 10 networks learnt the task;
  - at least 90% of those pass;
  - the positive control passes on at least 90% of its runs, and each negative control fails on at least 90% of its runs;
  - the Benjamini-Hochberg q over every claim in this ledger is below 0.10.

The thresholds were set by looking at the dev networks (modular addition seed 0, the chain engine seed 10) and nothing else. One claim, `hidden-circuit`, came from that look: the dev memorizer, wrong on 94% of the held-out pairs, was right on every pair once each neuron was cut down to its top cosine. It was declared and tested like the others, on ten memorizers nothing had looked at.

## What this is not

- Two small networks with clean tasks. Modular addition was chosen because its answer is known; most networks' mechanisms are not this tidy, and nothing here shows the probes would find one in a large model.
- The probes are written for these architectures (a Fourier basis for a cyclic task, ablation and patching for a pointer). Pointing the engine at a new network means writing probes for it, then declaring claims about it.
- The rebuild's template (a quadratic network whose neurons are cosines) is code. The facts supply the frequencies, the phases and the law, and the controls check that they matter: a rebuild with the wrong law fails, and one from a network without the mechanism is refused. But the engine did not invent the template.

## A billion-parameter language model (N2)

```
pip install -e .[lm]                 # torch and transformers; weights download to <home>/hf
python -m brain neuro lm-declare     # pins each model's Hugging Face commit and the code's sha
git commit ...
python -m brain neuro lm-test        # Pythia-1B, Pythia-1.4B and their untrained step0 checkpoints, on CPU
python -m brain neuro lm-card
```

The same questions, asked of EleutherAI's Pythia-1B (1.01 billion parameters) and Pythia-1.4B (1.41 billion), open weights under Apache-2.0 (`brain/neuro/lm.py`, `brain/neuro/lmclaims.py`).

**The behaviour.** Both models continue `"23, 24, 25,"` with `" 26"`. Numbers up to 530 are single tokens in their tokenizer, so a model cannot copy digits: it must know which number follows the token 25. Two-digit addition was the first choice, but the dev model got 3% of sums right, and a claim about how a model adds means nothing if it cannot add.

**The known answer.** Language models hold a number as a helix: its value plus waves of period 2, 5, 10 and 100 (Zhou et al. 2024, [arXiv:2406.03445](https://arxiv.org/abs/2406.03445); Kantamneni & Tegmark 2025, [arXiv:2502.00873](https://arxiv.org/abs/2502.00873), whose subjects included Pythia-6.9B).

**Where to look.** Everything reads or edits the model's state at the token of the last number, at one layer per model. The layer is chosen before any claim is measured: it is the layer where pasting in the state that another prompt had at its number works best. That choice uses a pairing of prompts (k → k + 61) that no claim uses.

**Looked at only on Pythia-410M** (`--trial` runs it):

| On the dev model, layer 8 | Helix | 4 other periods | Untrained step0 |
| --- | --- | --- | --- |
| R² on numbers held out of the fit | 0.40 | about 0.06 | −0.12 |
| Number read back exactly from the helix part | 0.97 | 0.00–0.07 | 0.01 |
| Logit gap toward m + 1 recovered by shifting only the helix part | 0.64 | 0.23 | – |
| Model then writes exactly m + 1 | 0.03 | 0.00 | – |

The real state of another prompt's number, pasted in at layer 8, redirects the count on 99% of prompts. After layer 13 the number has moved on to the next token, and pasting at the number token does nothing.

**The claims, for each of the two models:**
- `helix-fit`: the helix predicts held-out numbers' states (R² ≥ 0.2).
- `helix-read`: the number decodes exactly from the helix part (≥ 90%).
- `locate`: the number is read from that token at that layer (pasting redirects ≥ 90%).
- `helix-steer`: moving only the helix part recovers at least half the logit gap, more than other periods.
- `helix-write`: that move alone makes the model write exactly m + 1 (≥ 50%).

`helix-write` is expected to fail. On the dev model the helix moved the model off its answer, but almost never onto the new one. It is declared so the negative is on the record.

Controls:
- For `helix-fit` and `helix-read`: planted helix states (positive) and the untrained checkpoint (negative).
- For `locate`: the same paste at the last layer (negative).
- For `helix-steer` and `helix-write`: a helix fitted to shuffled numbers (negative).

The false-discovery correction is shared with the N1 claims.

**What it would and would not show.** If `helix-read` passes and `helix-write` fails, the bank's description of how a billion-parameter model holds a number is enough to read the number out of it exactly. It is not enough to write one in: the model also uses directions the helix leaves out. That is a description of one representation at one layer for one task. It says nothing about how the rest of the model works.
