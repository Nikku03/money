"""The scan explorer: the latest scan, the latest check and the latest desk note as one table, for a page built on
Perspective (an open-source data grid and pivot table, https://perspective-dev.github.io). docs/EXPLORER.md.

``python -m brain explorer build`` writes ``var/explorer/data.json``: one row per stock of the latest recorded scan
(``research/scans/<date>.json``) with its company and SEC industry, its ranges and its chance of a 20%+ day, why
its range is wide (the scan's own reasons, cached per scan), and, from the latest check against that scan (the desk's
case file), its move since the close, where that move sits against its ranges and why it moved. Only numbers the
repository already records go in it (ranges, chances, moves in percent); no price, volume or other Yahoo data.

``python -m brain explorer vendor`` puts Perspective's browser files (pinned below) in ``var/explorer/lib`` with
``npm pack``, laid out as their packages are, so the files find each other's WASM by relative path. The page
(``web/explorer/index.html``) and those files are published together; the page falls back to its own plain table
if Perspective cannot start.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from brain import config
from brain.bank import Bank, canon
from brain.timeutil import now_iso

PERSPECTIVE = "5.6.1"       # published 2026-10-23; pinned (a version at least two weeks old when chosen)
PACKAGES = ("client", "server", "viewer", "viewer-datagrid", "viewer-charts")
KEEP = {            # the files the page loads, per package, as the package lays them out
    "client": ("dist/cdn/perspective.js", "dist/cdn/perspective-server.worker.js", "dist/wasm/perspective-js.wasm"),
    "server": ("dist/wasm/perspective-server.wasm", "dist/wasm/perspective-server.memory64.wasm"),
    "viewer": ("dist/cdn/perspective-viewer.js", "dist/wasm/perspective-viewer.wasm", "dist/css/pro.css",
               "dist/css/pro-dark.css"),
    "viewer-datagrid": ("dist/cdn/perspective-viewer-datagrid.js",),
    "viewer-charts": ("dist/cdn/perspective-viewer-charts.js",),
}
COLUMNS = ("Ticker", "Company", "Industry", "Sector", "Move now %", "Where now", "2-in-3 low %", "2-in-3 high %",
           "9-in-10 low %", "9-in-10 high %", "Typical move %", "20% day chance %", "Listed after 2010",
           "Why the range is wide", "Why it moved")
INSIDE, OUT68, OUT90, NONE = "inside both ranges", "outside 2-in-3", "outside 9-in-10", "no reading"


def home() -> Path:
    return config.home() / "explorer"


def page() -> Path:
    return config.ROOT / "web" / "explorer" / "index.html"


# ---------- reading what the repository recorded ----------
def latest_scan() -> tuple[Path, dict] | None:
    from brain import scan
    files = sorted(scan.scans_dir().glob("????-??-??.json"))
    return (files[-1], json.loads(files[-1].read_text())) if files else None


def latest_case(asof: str) -> dict | None:
    """The desk's latest case file built against the scan of ``asof`` (it holds every checked stock's move)."""
    d = config.home() / "desk" / asof
    files = sorted(d.glob("*.json")) if d.exists() else []
    return json.loads(files[-1].read_text()) if files else None


def latest_update(asof: str, check: str | None) -> dict | None:
    from brain import scan
    d = scan.scans_dir() / "updates" / asof
    p = d / f"{check}.json" if check else None
    if p is None or not p.exists():
        files = sorted(d.glob("*.json")) if d.exists() else []
        p = files[-1] if files else None
    return json.loads(p.read_text()) if p else None


def latest_note(asof: str) -> tuple[Path, dict] | None:
    d = config.ROOT / "research" / "desk" / asof
    files = sorted(d.glob("*.json")) if d.exists() else []
    return (files[-1], json.loads(files[-1].read_text())) if files else None


def _rel(p: Path) -> str:
    """A path as the repository names it, or as it is when it lies elsewhere (a test's scratch folder)."""
    try:
        return p.resolve().relative_to(config.ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)


def _pct(x, digits: int = 1):
    return None if x is None or not math.isfinite(x) else round(float(x) * 100, digits)


def where(move: float | None, lo68: float, hi68: float, lo90: float, hi90: float) -> str:
    """Where a move since the close sits against the scan's ranges (simple returns), as the check reads it."""
    if move is None or not math.isfinite(move):
        return NONE
    if move < lo90 or move > hi90:
        return OUT90
    if move < lo68 or move > hi68:
        return OUT68
    return INSIDE


def market_line(market: list[dict]) -> str:
    return ", ".join(f"{m['name']} {m['move'] * 100:+.1f}%" for m in market if m.get("move") is not None)


# ---------- why each range is wide (the scan's own reasons, once per scan) ----------
def reasons(bank: Bank, asof: str, tickers: list[str], log=None) -> dict[str, str]:
    """The scan's reasons for each stock's range at the close of ``asof``, cached in ``var/explorer``. Empty when the
    bank's latest close is no longer ``asof`` (the reasons would describe another day)."""
    p = home() / f"why-{asof}.json"
    if p.exists():
        return json.loads(p.read_text())
    from brain import scan
    model = scan.load()
    if model is None:
        return {}
    res = scan.live(bank, tickers, log=log)
    if res["asof"] != asof:
        return {}
    known = scan.report_dates(asof, [r["t"] for r in res["rows"]])
    out = {r["t"]: "; ".join(scan.reasons(model["w1"], r, scan.report_note(r, known.get(r["t"]))))
           or "nothing unusual: its ordinary swings" for r in res["rows"]}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(canon(out) + "\n")
    return out


# ---------- the table ----------
def table(recorded: dict, ind: dict[str, dict], why_wide: dict[str, str], moves: dict[str, float],
          flags: dict[str, dict]) -> dict[str, list]:
    """Column-oriented rows (Perspective's preferred shape), one per recorded stock."""
    from brain.why import SIC_MAJOR
    newer = set(recorded.get("newer_listings") or [])
    cols: dict[str, list] = {c: [] for c in COLUMNS}
    for t in sorted(recorded["rows"]):
        lo90, lo68, hi68, hi90, shock = recorded["rows"][t]
        lo90, lo68, hi68, hi90 = (math.expm1(x) for x in (lo90, lo68, hi68, hi90))
        i = ind.get(t) or {}
        mv = moves.get(t)
        mv = None if mv is None or not math.isfinite(mv) else float(mv)
        f = flags.get(t) or {}
        moved = f.get("why") or ""
        if f.get("headlines"):
            moved = (moved + "; " if moved else "") + "headline: " + str(f["headlines"][0])[:160]
        cols["Ticker"].append(t)
        cols["Company"].append(i.get("company") or "")
        cols["Industry"].append(i.get("name") or "")
        cols["Sector"].append(SIC_MAJOR.get((i.get("sic") or "")[:2], "") if i.get("sic") else "")
        cols["Move now %"].append(_pct(mv, 2))
        cols["Where now"].append(where(mv, lo68, hi68, lo90, hi90))
        cols["2-in-3 low %"].append(_pct(lo68))
        cols["2-in-3 high %"].append(_pct(hi68))
        cols["9-in-10 low %"].append(_pct(lo90))
        cols["9-in-10 high %"].append(_pct(hi90))
        cols["Typical move %"].append(_pct((hi68 - lo68) / 2))
        cols["20% day chance %"].append(_pct(shock) if shock is not None else None)
        cols["Listed after 2010"].append("yes" if t in newer else "no")
        cols["Why the range is wide"].append(why_wide.get(t, ""))
        cols["Why it moved"].append(moved)
    return cols


def build(bank: Bank | None = None, out: Path | None = None, log=print) -> Path:
    """Write ``data.json`` for the page from the latest scan, its latest check and its latest desk note."""
    from brain import why
    bank = bank or Bank()
    got = latest_scan()
    if got is None:
        raise SystemExit("no recorded scan: run python -m brain scan run --refresh --record")
    path, recorded = got
    asof = recorded["asof"]
    tickers = sorted(recorded["rows"])
    case = latest_case(asof)
    upd = latest_update(asof, case["check"] if case else None)
    note = latest_note(asof)
    ind = why.industry_map(bank.view(), tickers)
    wide = reasons(bank, asof, tickers, log=log)
    moves = (case or {}).get("moves") or {}
    flags = {f["t"]: f for f in (upd or {}).get("flags") or []}
    data = {
        "made_at": now_iso(),
        "scan": {"asof": asof, "file": _rel(path), "stocks": len(tickers),
                 "left_out": recorded.get("skipped"), "newer": len(recorded.get("newer_listings") or [])},
        "check": None if upd is None else {
            "at": upd["at"], "session": (upd.get("market") or [{}])[0].get("session"),
            "line": market_line(upd.get("market") or []), "fear_up": bool(upd.get("fear_up")),
            "market_wide": bool(upd.get("market_wide")), "checked": upd.get("checked"),
            "outside68": upd.get("outside68"), "outside90": upd.get("outside90")},
        "note": None if note is None else {
            "file": _rel(note[0]), "check": note[1].get("check"),
            "themes": [{"name": th.get("name"), "kind": th.get("kind"), "members": th.get("members") or []}
                       for th in (note[1].get("note") or {}).get("themes") or []],
            "singles": [s.get("t") for s in (note[1].get("note") or {}).get("singles") or []]},
        "columns": list(COLUMNS),
        "rows": table(recorded, ind, wide, moves, flags),
    }
    out = out or home() / "data.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")
    if log:
        n_moves = sum(v is not None for v in data["rows"]["Move now %"])
        log(f"wrote {out}: {len(tickers)} stocks from the scan of {asof}, {n_moves} with a move from the check "
            f"{(case or {}).get('check')}, {sum(bool(x) for x in data['rows']['Why the range is wide'])} with reasons")
    return out


# ---------- Perspective's browser files ----------
def _safe_extract(tgz: Path, dest: Path) -> None:
    with tarfile.open(tgz) as tf:
        for m in tf.getmembers():
            target = (dest / m.name).resolve()
            if not str(target).startswith(str(dest.resolve())) or m.issym() or m.islnk():
                raise SystemExit(f"refusing an odd path in {tgz.name}: {m.name}")
        tf.extractall(dest, filter="data")


def vendor(dest: Path | None = None, version: str = PERSPECTIVE, log=print) -> list[Path]:
    """``npm pack`` each package at ``version`` and keep the files the page loads under ``dest/lib/<package>/``."""
    dest = dest or home()
    if shutil.which("npm") is None:
        raise SystemExit("npm is needed to fetch Perspective's files")
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in PACKAGES:
            spec = f"@perspective-dev/{name}@{version}"
            r = subprocess.run(["npm", "pack", "--silent", spec], cwd=tmp, capture_output=True, text=True, check=False)
            if r.returncode != 0:
                raise SystemExit(f"npm pack {spec} failed: {r.stderr.strip()[:300]}")
            tgz = tmp / r.stdout.strip().splitlines()[-1]
            x = tmp / name
            x.mkdir()
            _safe_extract(tgz, x)
            for rel in KEEP[name]:
                src = x / "package" / rel
                if not src.exists():
                    raise SystemExit(f"{spec} has no {rel}")
                dst = dest / "lib" / name / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                out.append(dst)
    if log:
        log(f"kept {len(out)} files of Perspective {version} in {dest / 'lib'} "
            f"({sum(p.stat().st_size for p in out) / 1e6:.1f} MB)")
    return out


def files_map(dest: Path | None = None) -> dict[str, str]:
    """The published path of every file the page needs, mapped to its local file (for the Artifact publish)."""
    dest = dest or home()
    root = config.ROOT.resolve()
    out = {"data.json": (dest / "data.json").resolve().relative_to(root).as_posix()}
    for name, rels in KEEP.items():
        for rel in rels:
            out[f"lib/{name}/{rel}"] = (dest / "lib" / name / rel).resolve().relative_to(root).as_posix()
    return out


def _cmd(a) -> int:
    if a.action == "vendor":
        vendor()
    elif a.action == "files":
        print(json.dumps(files_map(), indent=1))
    else:
        build()
    return 0


def add_cli(sub) -> None:
    p = sub.add_parser("explorer", help="the scan explorer page's data and files (docs/EXPLORER.md)")
    p.add_argument("action", choices=["build", "vendor", "files"])
    p.set_defaults(fn=_cmd)

