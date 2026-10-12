import os
import tempfile
import unittest

import pandas as pd

from brain import series
from brain.bank import Bank, Fact


class TempHome(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.environ.get("MONEY_HOME")
        os.environ["MONEY_HOME"] = self._tmp.name
        self.bank = Bank()

    def tearDown(self):
        if self._old is None:
            os.environ.pop("MONEY_HOME", None)
        else:
            os.environ["MONEY_HOME"] = self._old
        self._tmp.cleanup()


def filed(entity, attr, value, as_of, known_at, **ctx):
    return Fact(entity=entity, attr=attr, value=value, as_of=as_of, known_at=known_at, tier="filed",
                source={"name": "test", "url": "https://example.test/" + attr}, context=ctx)


def price_snapshot(entity, dates, close, adj=None, splits=()):
    df = pd.DataFrame({"date": dates, "open": close, "high": close, "low": close, "close": close,
                       "adjclose": adj if adj is not None else close, "volume": [1] * len(dates)})
    sha = series.store(df)
    return Fact(entity=entity, attr="series.price_daily", value=sha, as_of=dates[-1], known_at=dates[-1],
                tier="market", source={"name": "test", "url": "https://example.test/px"},
                context={"rows": len(dates), "first": dates[0], "last": dates[-1], "splits": [list(s) for s in splits]})
