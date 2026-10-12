"""Paths and identity for outbound requests."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def home() -> Path:
    """Local state (journal, cache, series, briefs). Gitignored."""
    p = Path(os.environ.get("MONEY_HOME", ROOT / "var"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def sec_user_agent() -> str:
    """SEC fair-access policy: requests must name who is asking."""
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua:
        raise SystemExit(
            "SEC EDGAR requires a User-Agent with contact details.\n"
            "  export SEC_USER_AGENT='Your Name your@email.com'"
        )
    return ua


WEB_USER_AGENT = "Mozilla/5.0 (compatible; money-brain/0.1)"
LEDGER = ROOT / "research" / "ledger.jsonl"
RESULTS = ROOT / "research" / "RESULTS.md"
UNIVERSE = ROOT / "universe" / "us_large.json"
