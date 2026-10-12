"""Timestamps. known_at is always full UTC 'YYYY-MM-DDTHH:MM:SSZ'.

A date-only known_at is taken as the end of that UTC day: a filing dated
2024-11-01 may have landed after the close, so it is not usable before then.
"""
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def norm_date(s) -> str:
    if isinstance(s, datetime):
        return s.astimezone(timezone.utc).strftime("%Y-%m-%d")
    if isinstance(s, date):
        return s.isoformat()
    s = str(s).strip()
    date.fromisoformat(s[:10])  # validates
    return s[:10]


def norm_known_at(s) -> str:
    if isinstance(s, datetime):
        dt = s if s.tzinfo else s.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    s = str(s).strip()
    if len(s) == 10:
        date.fromisoformat(s)
        return s + "T23:59:59Z"
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rfc822_to_iso(s: str) -> str:
    return norm_known_at(parsedate_to_datetime(s))


def days_between(a: str, b: str) -> int:
    return (date.fromisoformat(b[:10]) - date.fromisoformat(a[:10])).days
