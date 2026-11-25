"""Counts only: which forms of guidance the research universe's releases carry. No outcome is read.

    PYTHONPATH=. python research/promises/survey_forms.py > research/promises/survey_forms.log

The patterns are loose (a revenue word or an earnings-per-share word near a forward word and a figure), so each
count is an upper bound on what a careful reader would keep. It sizes the next layer of the promise record
(docs/FILE.md, "What it does not hold yet")."""
import json, re, sys
from collections import defaultdict
from brain import config, promises
from brain.bank import Bank

FWD = promises._FWD
NUM = r"-?\d+(?:\.\d+)?"
PCT_RANGE = re.compile(NUM + r"\s?%?\s*(?:to|and|-|–|—)\s*" + NUM + r"\s?(?:%|percent)", re.I)
PCT_ONE = re.compile(NUM + r"\s?(?:%|percent)", re.I)
REV = re.compile(r"\b(?:total\s+|net\s+|organic\s+)?(?:revenues?|sales)\b", re.I)
GROW = re.compile(r"\b(growth|grow|increase|increases|up|decline|declines|decrease|down)\b", re.I)
EPS = re.compile(r"\b(?:earnings per (?:diluted )?share|EPS|net income per (?:diluted )?share)\b", re.I)
DOLLAR_RANGE = re.compile(r"\$\s?(\d+\.\d{2})\s*(?:to|and|-|–|—)\s*\$?\s?(\d+\.\d{2})")
DOLLAR_ONE = re.compile(r"\$\s?(\d+\.\d{2})\b(?!\s?(?:million|billion|thousand))")

def forms(text):
    got = set()
    for m in REV.finditer(text):
        seg = text[m.end():m.end() + 140]
        lead = text[max(0, m.start() - 200):m.start()]
        if not (FWD.search(seg) or FWD.search(lead)):
            continue
        r = PCT_RANGE.search(seg)
        if r and GROW.search(text[max(0, m.start() - 60):m.end() + r.end()]) and "$" not in seg[:r.start()]:
            got.add("rev_growth_range")
        elif (o := PCT_ONE.search(seg)) and GROW.search(text[max(0, m.start() - 60):m.end() + o.end()]) \
                and "$" not in seg[:o.start()]:
            got.add("rev_growth_point")
    for m in EPS.finditer(text):
        seg = text[m.end():m.end() + 140]
        lead = text[max(0, m.start() - 200):m.start()]
        if not (FWD.search(seg) or FWD.search(lead)):
            continue
        if DOLLAR_RANGE.search(seg):
            got.add("eps_range")
        elif DOLLAR_ONE.search(seg):
            got.add("eps_point")
    return got

names = sorted({t.upper() for t in json.loads((config.ROOT / "universe" / "us_broad.json").read_text())["tickers"]})
view = Bank().view()
lib = [r for r in promises.load() if r["t"] in set(names)]
dollar = defaultdict(set)
for r in lib:
    dollar["all" if r["made"] < "2024" else "late"].add(r["t"])
    dollar["any"].add(r["t"])
cnt = defaultdict(lambda: defaultdict(set)); rel = defaultdict(lambda: defaultdict(int)); total = defaultdict(int)
ex = defaultdict(list)
for t in names:
    for f in view.facts(t, promises.ATTR):
        c = f.context or {}
        if c.get("reader", 1) < 3:
            continue
        era = "2011-2023" if f.known_at[:4] < "2024" else "2024-2026"
        total[era] += 1
        text = c.get("outlook") or ""
        for k in forms(text):
            cnt[era][k].add(t); rel[era][k] += 1
for era in ("2011-2023", "2024-2026"):
    print(f"\n{era}: {total[era]} releases read with reader 3")
    for k in ("rev_growth_range", "rev_growth_point", "eps_range", "eps_point"):
        print(f"  {k:<18} {rel[era][k]:>6} releases, {len(cnt[era][k]):>4} companies")
    g = cnt[era]["rev_growth_range"] | cnt[era]["rev_growth_point"]
    e = cnt[era]["eps_range"] | cnt[era]["eps_point"]
    d = {r["t"] for r in lib if (r["made"][:4] < "2024") == (era == "2011-2023")}
    print(f"  companies with a dollar revenue promise (the library): {len(d)}")
    print(f"  with revenue growth guidance but no dollar promise: {len(g - d)}")
    print(f"  with EPS guidance (range or point): {len(e)}; of them no dollar revenue promise: {len(e - d)}")
    print(f"  any of the three: {len(d | g | e)} of {len(names)}")
