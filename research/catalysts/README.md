# Catalyst calls

Probabilities written down **before** a clinical readout, then scored. Each line of `calls.jsonl` is one of:

- **`call`:** a yes/no question with a threshold anyone can check against the presentation, the probability of yes, what was already public, why, and the sources read.
- **`outcome`:** what happened, citing the page that settles it.
- **`void`:** a badly posed call, kept with its reason.

The commit that adds a call is the proof it came before the event. Nothing is edited or deleted.

```
python -m brain catalysts list
python -m brain catalysts resolve 2026-10-VERA-ORIGIN3 yes --source https://... [--note "..."]
python -m brain catalysts score      # Brier score; always saying 50% scores 0.25
```

## What a call means

- These are an agent's judgments from public sources, with no track record behind them yet. Like any source's claims (CLAUDE.md), they carry weight only once `score` shows a record long enough to mean something.
- Most October 2026 readouts had already put out a topline press release. So most calls ask about magnitude (a hazard ratio threshold), not pass or fail.
- A high probability on an already-announced result is not an edge. The market knew the topline too.
