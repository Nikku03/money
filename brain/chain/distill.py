"""Can the chain engine's knowledge come from a language model alone? (docs/LINKS.md, "Links from a model")

The model is given only the names of the market states in ``knowledge/links.json``: no links, no data. State
by state, it is asked what that state directly causes within about a month, and in which direction. Its answers
become a link table of its own (``<home>/knowledge/links.<model>.json``). Then three questions:

1. **Recall.** How many of the textbook links did it write on its own, with the right sign, and what did it add?
2. **Reasoning.** Can the chain engine follow chains through the model's table? Each chain is checked against an
   exact lookup.
3. **Reality.** For each cause -> basket pair the model's table implies and the bank can measure, its direction
   is set beside the link tests already recorded (research/RESULTS.md, Links). Nothing is measured here: a pair
   the tests have not covered stays untested until it is declared.

The table is the model's opinion, like a web page's claim. It carries no weight anywhere.
"""
from __future__ import annotations

import hashlib
import json
import re

from brain import config
from brain.timeutil import now_iso

PROMPT = (
    "Market state: {cause}.\n"
    "Which of the states below does it directly cause within about a month? Only direct effects (one step), not "
    "chains. Leave out anything it does not clearly cause.\n{states}\n"
    "For each effect give the state's exact name and its sign: 1 if the cause makes that state happen as named, "
    "-1 if it makes the opposite happen.\n"
    'Answer with JSON only: {{"effects": [{{"state": "exact_name", "sign": 1}}]}}. An empty list is fine. /no_think'
)
SYSTEM = "You are an economist. You answer with JSON only."
LLM_FIXED = {"temperature": 0, "seed": 7, "max_tokens": 600}
MAX_CHAINS = 400     # chains the engine follows per run, in the lookup's order; the rest are counted, not run


def states(links: list[dict]) -> list[str]:
    """Every state the textbook links name, in first-seen order."""
    out: dict[str, None] = {}
    for ln in links:
        out.setdefault(ln["cause"], None)
        out.setdefault(ln["effect"], None)
    return list(out)


def prompt(cause: str, names: list[str]) -> str:
    listed = "\n".join(f"- {n}" for n in names if n != cause)
    return PROMPT.format(cause=cause, states=listed)


def prompt_sha() -> str:
    return hashlib.sha256((SYSTEM + "\n" + PROMPT).encode()).hexdigest()[:16]


def parse(text: str, cause: str, names: list[str]) -> tuple[list[dict], str]:
    """(effects, "") or ([], why not). Keeps known names other than the cause, signs 1 or -1, first one wins."""
    from brain.judge import clean
    s = clean(text)
    if "{" not in s:
        return [], "the answer holds no JSON object"
    try:
        obj = json.loads(s[s.find("{"): s.rfind("}") + 1])
    except ValueError:
        return [], "the answer's JSON does not parse"
    items = obj.get("effects") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return [], 'the answer has no "effects" list'
    known, out, seen = set(names), [], set()
    for it in items:
        if not isinstance(it, dict):
            continue
        name = str(it.get("state", "")).strip().strip("`'\" ").lower().replace(" ", "_")
        sign = it.get("sign")
        if isinstance(sign, str) and re.fullmatch(r"[+-]?1", sign.strip()):
            sign = int(sign.strip())
        if name in known and name != cause and sign in (1, -1) and name not in seen:
            seen.add(name)
            out.append({"cause": cause, "effect": name, "sign": int(sign)})
    return out, ""


def _slug(model: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", model.lower()).strip("-") or "model"


def path_for(model: str):
    return config.home() / "knowledge" / f"links.{_slug(model)}.json"


def ask(client, links: list[dict], log=print) -> dict:
    """The model's own link table, asked state by state; saved under the home and returned."""
    names = states(links)
    out, failed = [], []
    for cause in names:
        msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(cause, names)}]
        try:
            reply = client.complete(msg, temperature=LLM_FIXED["temperature"], seed=LLM_FIXED["seed"],
                                    max_tokens=LLM_FIXED["max_tokens"])["choices"][0]["message"].get("content") or ""
        except Exception as e:  # noqa: BLE001 - one failed answer is recorded, the rest still run
            failed.append({"cause": cause, "why": f"{type(e).__name__}: {e}"})
            continue
        effects, why = parse(reply, cause, names)
        if why:
            failed.append({"cause": cause, "why": why})
        out += effects
        if log:
            said = ", ".join(f"{e['effect']} {e['sign']:+d}" for e in effects)
            log(f"  {cause}: {said or 'nothing'}"
                + (f" ({why})" if why else ""))
    table = {"about": "Cause -> effect links written by a language model from the state names alone (brain/chain/"
                      "distill.py). The model's opinion: untested, and weighs nothing.",
             "model": getattr(client, "model", "?"), "prompt_sha": prompt_sha(), "llm": LLM_FIXED, "made_at": now_iso(),
             "states": names, "failed": failed,
             "links": [{**ln, "weight": 1.0, "source": f"model {getattr(client, 'model', '?')}", "status": "untested"}
                       for ln in out]}
    p = path_for(table["model"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(table, indent=1))
    table["path"] = str(p)
    return table


def compare(model_links: list[dict], textbook: list[dict], results: dict, net=None) -> dict:
    """Recall against the textbook, the chain engine on the model's table, and the model's measurable pairs
    beside the recorded link tests (``results``: verdict rows by id, from ``linktest.status``)."""
    from brain import chain
    from brain.chain import linktest
    tb = {(ln["cause"], ln["effect"]): int(ln.get("sign", 1)) for ln in textbook}
    md = {(ln["cause"], ln["effect"]): int(ln.get("sign", 1)) for ln in model_links}
    found = [k for k in tb if k in md]
    recall = {"textbook": len(tb), "found": len(found), "same_sign": sum(md[k] == tb[k] for k in found),
              "missed": [f"{c} -> {e}" for (c, e) in tb if (c, e) not in md],
              "wrong_sign": [f"{c} -> {e}" for (c, e) in found if md[(c, e)] != tb[(c, e)]],
              "added": [f"{c} -> {e} ({s:+d})" for (c, e), s in md.items() if (c, e) not in tb]}
    reasoning = {"chains": 0, "followed": 0, "agree": 0}
    if net is not None:
        for start in dict.fromkeys(ln["cause"] for ln in model_links):
            n = len(chain.paths(model_links, start, chain.M - 1))
            reasoning["chains"] += n
            room = MAX_CHAINS - reasoning["followed"]
            if room <= 0:
                continue
            rows = chain.follow(net, model_links, start, chain.M - 1)[:room]
            reasoning["followed"] += len(rows)
            reasoning["agree"] += sum(r["agree"] for r in rows)
    reality, seen = [], set()
    tb_pairs = {(p["cause"], p["effect"]): p["sign"] for p in linktest.pairs(textbook)}
    for p in linktest.pairs(model_links):
        key = (p["cause"], p["effect"])
        if key in seen:
            continue  # one row per pair; the first path stands for it
        seen.add(key)
        row = {"pair": f"{p['cause']} -> {p['effect']}", "model_sign": p["sign"], "textbook_sign": tb_pairs.get(key),
               "path": " -> ".join([p["path"][0]["cause"]] + [s["effect"] for s in p["path"]])}
        same, nxt = results.get(linktest.test_id(p, "same")), results.get(linktest.test_id(p, "next"))
        declared = tb_pairs.get(key)   # the recorded tests' one-sided p is in the textbook's direction
        if same is None or declared is None:
            row["same_month"] = "untested: a new pair, measured only once declared"
        else:
            te = same["metrics"]["test"]
            p_model = te["p"] if p["sign"] == declared else 1 - te["p"]   # the same test, in the model's direction
            word = ("the data agrees" if p_model < 0.05 else "the data says the opposite" if p_model > 0.95
                    else f"the data cannot tell (p {p_model:.2f})")
            row["same_month"] = f"{word}: {te['per_move'] * 100:+.2f} points per move"
        if nxt is None or declared is None:
            row["next_month"] = "untested"
        else:
            row["next_month"] = nxt["verdict"] if p["sign"] == declared else f"tested only the other way: {nxt['verdict']}"
        reality.append(row)
    return {"recall": recall, "reasoning": reasoning, "reality": reality}


def report(table: dict, cmp: dict) -> str:
    r, g = cmp["recall"], cmp["reasoning"]
    lines = [(f"Model {table['model']} wrote {len(table['links'])} links from {len(table['states'])} state names alone"
              + (f" ({len(table['failed'])} answers unusable)" if table["failed"] else "") + f": {table.get('path', '')}"),
             "",
             (f"1. Recall: {r['found']} of the {r['textbook']} textbook links, {r['same_sign']} with the right sign; "
              f"{len(r['added'])} links the textbook does not have.")]
    if r["wrong_sign"]:
        lines.append(f"   wrong sign: {'; '.join(r['wrong_sign'])}")
    if r["missed"]:
        lines.append(f"   missed: {'; '.join(r['missed'])}")
    if r["added"]:
        lines.append(f"   added: {'; '.join(r['added'][:20])}" + (" ..." if len(r["added"]) > 20 else ""))
    lines.append(f"2. Reasoning: the chain engine followed {g['followed']} of {g['chains']} chains through the model's "
                 f"table; {g['agree']} agree with an exact lookup.")
    lines.append(f"3. Reality: {len(cmp['reality'])} cause -> basket pairs the bank can measure.")
    for row in cmp["reality"]:
        tb = {None: "the textbook has no such pair", 1: "textbook: up", -1: "textbook: down"}[row["textbook_sign"]]
        lines.append(f"   {row['pair']}: model says {'up' if row['model_sign'] > 0 else 'down'} ({tb}). Same month: "
                     f"{row['same_month']}. Next month: {row['next_month']}.")
    lines.append("The model's table is its opinion: it weighs nothing, and a new pair counts only once a declared "
                 "link test has measured it.")
    return "\n".join(lines)
