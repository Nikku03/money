"""World radar: which markets moved hard. Reads the bank's daily closes for the watched world markets
(brain/markets.py) and a list of US-listed country funds, and flags a symbol whose move over a day, a week
or a month, or whose fall from its three-month high, passes its kind's threshold.

It reports what already happened, sharply and early; it predicts nothing, and no flag has been tested as a
signal. Country funds come from the same chart source as equities, under the same terms (personal research,
snapshots stay in the gitignored home).
"""
from __future__ import annotations

from brain import markets, series
from brain.bank import Bank

# US-listed country funds: the cheapest daily read on markets the index list does not carry.
COUNTRIES = [("EWZ", "Brazil"), ("EWY", "South Korea"), ("INDA", "India"), ("EIDO", "Indonesia"),
             ("EWT", "Taiwan"), ("EWW", "Mexico"), ("EZA", "South Africa"), ("TUR", "Turkey"),
             ("EWJ", "Japan"), ("FXI", "China large caps"), ("EEM", "Emerging markets"), ("ARGT", "Argentina"),
             ("ECH", "Chile"), ("EPU", "Peru"), ("THD", "Thailand"), ("EWM", "Malaysia"), ("EPHE", "Philippines"),
             ("VNM", "Vietnam"), ("KSA", "Saudi Arabia"), ("EPOL", "Poland"), ("EWG", "Germany"),
             ("EWU", "United Kingdom"), ("EWA", "Australia"), ("EWC", "Canada")]
COUNTRY_NAMES = dict(COUNTRIES)

# Percent thresholds per kind: one day, five days, 21 days, fall from the 63-day high.
LIMITS = {"index": (3, 7, 10, 15), "fund": (3, 7, 10, 15), "fx": (1.5, 3, 5, 8), "commodity": (4, 10, 15, 20),
          "crypto": (8, 15, 25, 35), "volatility": (20, 40, 60, 99)}
STALE_DAYS = 5


def watched() -> list[dict]:
    rows = [{"symbol": m["symbol"], "name": m["name"], "kind": m["kind"]} for m in markets.MARKETS]
    return rows + [{"symbol": s, "name": n, "kind": "fund"} for s, n in COUNTRIES]


def moves(closes: list[float]) -> dict | None:
    """Percent moves over 1, 5 and 21 sessions and the fall from the 63-session high, from daily closes."""
    if len(closes) < 22:
        return None
    last = closes[-1]
    pct = lambda n: (last / closes[-1 - n] - 1) * 100
    return {"d1": pct(1), "d5": pct(5), "d21": pct(21), "from_high": (last / max(closes[-63:]) - 1) * 100}


def flags(kind: str, m: dict) -> list[str]:
    d1, d5, d21, dd = LIMITS.get(kind, LIMITS["index"])
    out = []
    for key, lim, label in (("d1", d1, "day"), ("d5", d5, "week"), ("d21", d21, "month")):
        if abs(m[key]) >= lim:
            out.append(f"{label} {m[key]:+.1f}%")
    if m["from_high"] <= -dd:
        out.append(f"{m['from_high']:.1f}% from 3-month high")
    return out


def scan(bank: Bank, today: str | None = None) -> list[dict]:
    """Every watched symbol the bank has closes for, with its moves, flags and a severity to sort by."""
    from datetime import date
    v = bank.view()
    out = []
    for w in watched():
        f = v.latest(w["symbol"], "series.price_daily")
        if f is None:
            out.append({**w, "missing": True})
            continue
        df = series.load(f.value)
        m = moves(df["close"].astype(float).tolist())
        if m is None:
            out.append({**w, "missing": True})
            continue
        last = str(df["date"].iloc[-1])
        stale = today is not None and (date.fromisoformat(today) - date.fromisoformat(last)).days > STALE_DAYS
        lim = LIMITS.get(w["kind"], LIMITS["index"])
        sev = max(abs(m["d1"]) / lim[0], abs(m["d5"]) / lim[1], abs(m["d21"]) / lim[2], -m["from_high"] / lim[3])
        out.append({**w, **{k: round(x, 2) for k, x in m.items()}, "last": last, "close": float(df["close"].iloc[-1]),
                    "flags": flags(w["kind"], m), "severity": round(sev, 2), "stale": stale})
    return sorted(out, key=lambda r: -r.get("severity", -1))


def refresh(bank: Bank, log=print) -> dict:
    """Re-read the world markets and the country funds' daily closes."""
    from brain.sources import prices
    out = markets.ingest(bank, log=log)
    for s, _ in COUNTRIES:
        try:
            out["added"][s] = bank.add_many(prices.ingest(s))
        except Exception as e:  # noqa: BLE001 - one fund's outage must not stop the scan
            out["failed"][s] = f"{type(e).__name__}: {e}"
            log(f"  {s}: skipped ({out['failed'][s]})")
    return out


def _cmd(a):
    from brain.timeutil import today
    bank = Bank()
    if a.refresh:
        refresh(bank, log=lambda m: None)
    rows = scan(bank, today())
    have = [r for r in rows if not r.get("missing")]
    if not have:
        print("No world-market closes in the bank: run `python -m brain radar --refresh`.")
        return 1
    hot = [r for r in have if r["flags"]]
    print(f"{len(hot)} of {len(have)} markets flagged (thresholds by kind: day / week / month / fall from "
          "3-month high). What happened, not a forecast.")
    for r in (have if a.all else hot):
        name = COUNTRY_NAMES.get(r["symbol"], r["name"])
        mark = "  STALE" if r["stale"] else ""
        print(f"  {r['symbol']:10} {name[:22]:22} {r['close']:>11.2f}  1d {r['d1']:+6.1f}%  5d {r['d5']:+6.1f}%  "
              f"1m {r['d21']:+6.1f}%  hi {r['from_high']:+6.1f}%  {', '.join(r['flags'])}{mark}  ({r['last']})")
    missing = [r["symbol"] for r in rows if r.get("missing")]
    if missing:
        print(f"No closes yet for: {', '.join(missing)} (`--refresh` reads them).")
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("radar", help="world radar: markets and country funds that moved hard (docs/PREDICT.md)")
    p.add_argument("--refresh", action="store_true", help="re-read world markets and country funds first")
    p.add_argument("--all", action="store_true", help="list every market, flagged or not")
    p.set_defaults(fn=_cmd)
