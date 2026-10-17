# Running the brain on Colab with an A100

On Colab, only one part needs the GPU: **the language model**, which reads and writes English. Everything else (the memory bank, the reasoning engine, the tests) runs on the CPU, exactly as it does here. The model is served on Colab by vLLM through an OpenAI-compatible endpoint, which is what `brain/llm.py` already speaks. So the code does not change: four environment variables point it at the GPU.

## 1. The big picture

```
                 ┌──────────────────────────── the open internet (public only) ────────────────────────────┐
                 │  SEC filings · prices · insider trades · macro · headlines · web pages · alt data       │
                 └───────────────┬──────────────────────────────────────────────┬──────────────────────────┘
                                 │ numbers                                       │ text
                                 ▼                                               ▼
                    ① COLLECTORS (CPU, no AI)                       ② READER (GPU, the language model)
                    brain ingest / insiders / macro /                brain scout · decide (judge) ·
                    markets / watch                                  [to build: event extraction]
                                 │                                               │ structured facts:
                                 │ facts with source + time                      │ claims, events, judgments
                                 ▼                                               ▼
                 ┌──────────────────────────── ③ MEMORY BANK (Google Drive) ───────────────────────────────┐
                 │  every fact: what, when it became public, where it came from · point-in-time views     │
                 │  [to build: ③b KNOWLEDGE BANK: cause→effect links from books/papers + case studies,     │
                 │   each link tested against the bank's history]                                         │
                 └───────────────┬────────────────────────────────────────────────────────────────────────┘
                                 ▼
                    ④ REASONING ENGINE (CPU, no AI: numbers and tests)
                    research (hypotheses, ledger, BH) · forecast F1 (learner, 49 variables, ranges)
                    decide D1 (weighs only evidence with a proven record)
                                 │ decision · forecast · cited evidence
                                 ▼
                    ⑤ EXPLAINER (GPU, the language model)
                    plain-words explanation · ASK analyst · terminal
                                 │
                                 ▼
                    ⑥ SCOREKEEPER (CPU)  ── 21 sessions later ──▶ track records ──▶ weights in ④
                    decide score · forecast score
```

## 2. Who does what

| # | Part | Its job | AI? | Runs on | Commands |
|---|---|---|---|---|---|
| 1 | Collectors | Fetch public numbers and filings; store each fact with its source and the time it became public | no | CPU | `brain ingest`, `ingest-universe --broad`, `insiders`, `macro`, `markets`, `alerts check` |
| 2 | Reader | Turn English into facts: a claim's direction, an event's type, the model's own call with the facts it cites | **yes** | **GPU** | `brain scout`, `brain decide` (judge), `brain forecast --llm` |
| 3 | Memory bank | Keep everything, append-only, point-in-time; nothing is overwritten | no | Drive (disk) | `brain check`, `brain show`, `brain ask` (a fact's history), `brain brief` |
| 3b | Knowledge bank *(to build)* | Cause→effect links ("oil +10% → airlines −?") from books and papers, plus case studies with their outcomes, each link scored on the bank's history | builds with AI, tests without | GPU to read, CPU to test | — |
| 4b | Chain engine (yours, from Nikku03/cell) | Follow cause→effect links step by step, exactly: the bump-node dendritic tree with the snap (`brain/chain`, links in `knowledge/links.json`). Every chain is shown beside an exact lookup; links count only once tested | no LLM; a small neural net | CPU (trains in about 20 s) | `brain chain train`, `brain chain follow oil_price_up` |
| 4 | Reasoning engine | Test ideas on held-out years; learn which variables matter; decide only on proven evidence; give price ranges | no | CPU | `brain research`, `brain forecast`, `brain decide` |
| 5 | Explainer | Say in plain words what the engine decided and why; answer questions, citing facts | **yes** | **GPU** | the decision card's explanation, `brain terminal` (its ASK pane) |
| 6 | Scorekeeper | After 21 sessions, check every decision, forecast and source against what the stock did; that record is what gives anything weight | no | CPU | `brain decide score`, `brain forecast score` |

The rule that ties it together: **the language model understands; the engine decides; the scorekeeper judges both.** A model's call is a source like any web page. It counts only after its scored record clears the gate (`docs/DECIDE.md`).

## 3. The daily loop

1. **After the US close** (about 21:00 UTC): collectors refresh prices, filings, insider trades and headlines (`brain ingest <the watchlist>` or `brain ingest-universe --broad`, then `brain alerts check`).
2. **The reader goes over what is new.** It reads that day's 8-Ks, earnings releases and headlines for the watchlist (`brain scout`) and stores claims, then makes its own call (`brain decide`).
3. **The reasoning engine** forecasts (`brain forecast`) and decides (`brain decide --watchlist`) from the bank only.
4. **The explainer** writes the card, and the terminal shows it.
5. **The scorekeeper** scores everything whose 21 sessions have passed. Records grow, and a source or model that earns a record starts to count.

## 4. Colab setup

**Storage.**
- Colab machines are wiped when the session ends. Keep the bank on Google Drive: `MONEY_HOME=/content/drive/MyDrive/money/var`.
- The bank is not in git (about 2.3 GB of facts and price series). Two ways to fill it:
  - **Rebuild on Colab:** `brain ingest-universe --broad`, `brain insiders`, `brain markets`, `brain macro`. Allow a few hours the first time; later days only fetch what is new.
  - **Copy the existing bank:** pack `var/bank` and `var/series` here (`tar czf bank.tgz var/bank var/series`) and upload the archive to Drive.

**The model** (served by vLLM on the same machine):

| GPU | Model | Weights | Context |
|---|---|---|---|
| A100 80 GB | `Qwen/Qwen3-32B` | BF16, about 65 GB | 32k |
| A100 40 GB | `Qwen/Qwen3-32B-AWQ` | 4-bit, about 20 GB | 16k-32k |
| A100 80 GB, if you want Llama | `meta-llama/Llama-3.3-70B-Instruct` in an AWQ build | 4-bit, about 40 GB | 16k |

- Qwen3 is the default: Apache-2.0, strong tool calling (the analyst and judge call tools), and the same family the code was tested with.
- Llama needs Meta's licence accepted on Hugging Face, and the `llama3_json` tool parser.
- There is no Llama 3 at 30B; the nearest are 8B and 70B.

**Serving** (one background cell):
```
vllm serve Qwen/Qwen3-32B --max-model-len 32768 --enable-auto-tool-choice --tool-call-parser hermes --port 8000
```

**Environment** (what the code reads):
```
SEC_USER_AGENT="Your Name your@email.com"
MONEY_HOME=/content/drive/MyDrive/money/var
BRAIN_LLM_URL=http://127.0.0.1:8000/v1
BRAIN_LLM_MODEL=Qwen/Qwen3-32B          # the id vLLM serves
BRAIN_LLM_CTX=32768
BRAIN_LLM_TIMEOUT=600
BRAIN_SEARX_URL=...                     # optional: a SearXNG instance for web search
```

**The terminal on Colab.** `brain terminal --no-browser --port 8765`, then open it through Colab's port proxy (`google.colab.output.serve_kernel_port_as_window(8765)`). The terminal only listens on the machine itself, and the proxy is what lets the browser reach it.

## 5. The notebook, cell by cell

1. **Check the GPU:** `nvidia-smi`; choose the model row from section 4.
2. **Mount Drive** and set `MONEY_HOME`.
3. **Get the code:** `git clone -b claude/upbeat-turing-z9jk2b https://github.com/nikku03/money`. A private repo needs a GitHub token. Then `pip install -e .`.
4. **Install vLLM, then start the model server** in the background; wait for `/v1/models`.
5. **Set the environment** (section 4), then run `python -m unittest discover -s tests -t .` and `brain check`.
6. **Data:** restore the bank from Drive, or rebuild it. Then the daily refresh.
7. **Read and decide:** `brain scout TICKER`, `brain decide --watchlist`, `brain forecast TICKER --llm`.
8. **Ask:** the terminal's ASK pane (`brain terminal`, through the port proxy): questions answered from the bank only, with citations.
9. **Score:** `brain decide score`, `brain forecast score`.
10. **Test the bigger model honestly** (section 6).
11. **Save:** the bank stays on Drive. The research ledger and RESULTS.md live in the repo, so commit and push them, or download them.

## 6. What the bigger model changes, and how to know

**It will read better:**
- fewer invented citations;
- sensible reasons;
- real use of the facts.

The 1.7B model tested here mostly answered 50/50 and once called a stock that trailed SPY "outperforming".

**Whether it predicts better is a separate question, and only a declared test answers it.**
- **Declare it as a new test.** Use the same 8-week, 12-stock retest as F1d, with the new model pinned. A new model is a new test id; the old result stays.
- **Commit before running,** as the research rules say.
- **Check the model's training cutoff.** It must predate the test window, or the model may already know what happened. Qwen3's training data ends before 2025-04, so the last two months of 2026 are safe.

## 7. To build next (in this order)

1. **Event extraction.**
   - The reader turns each new filing and headline into a labelled event: guidance raised or cut, lawsuit, launch, management change and so on.
   - It stores them in the bank with their sources.
   - The research engine then tests each event type against history.
2. **Knowledge bank (3b).** Cause→effect links and case studies, each scored on the bank's history, with failures as well as successes, so the engine can map new news to affected stocks with a measured, not assumed, effect.
3. **Expectations data.** Analyst estimates and options-implied moves. "Better than expected" is what moves prices, and the bank has no expectations today.
