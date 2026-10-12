"""Time series (prices, macro) live in content-addressed CSV files; the bank holds a
fact per snapshot whose value is the file's sha256. Same data, same file."""
from __future__ import annotations

import functools
import hashlib
import io

import pandas as pd

from brain import config


def _dir():
    d = config.home() / "series"
    d.mkdir(parents=True, exist_ok=True)
    return d


def store(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.to_csv(buf, index=False, float_format="%.6f", lineterminator="\n")
    data = buf.getvalue().encode()
    sha = hashlib.sha256(data).hexdigest()
    p = _dir() / f"{sha}.csv"
    if not p.exists():
        p.write_bytes(data)
    return sha


def path(sha: str):
    return _dir() / f"{sha}.csv"


@functools.lru_cache(maxsize=512)
def _load_cached(home: str, sha: str) -> pd.DataFrame:
    return pd.read_csv(path(sha), dtype={"date": str})


def load(sha: str) -> pd.DataFrame:
    """Snapshots are content-addressed and immutable, so caching by sha is safe. Returns a copy."""
    return _load_cached(str(config.home()), sha).copy()


def verify(sha: str) -> bool:
    p = path(sha)
    return p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == sha
