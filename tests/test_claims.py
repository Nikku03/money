import pandas as pd

from brain import claims
from tests.util import TempHome, price_snapshot


class ClaimsTest(TempHome):
    def test_claims_are_scored_against_the_market_after_publication(self):
        days = pd.bdate_range("2024-01-01", periods=60).strftime("%Y-%m-%d").tolist()
        up = [100 * 1.01 ** i for i in range(60)]     # beats a flat market
        flat = [100.0] * 60
        self.bank.add(price_snapshot("X", days, up))
        self.bank.add(price_snapshot("SPY", days, flat))
        claims.note(self.bank, "X", "X will rally", "https://good.example/a", direction="up", known_at=days[5])
        claims.note(self.bank, "X", "X will fall", "https://bad.example/b", direction="down", known_at=days[5])
        claims.note(self.bank, "X", "too recent", "https://bad.example/c", direction="down", known_at=days[50])
        claims.note(self.bank, "X", "no view", "https://good.example/d", known_at=days[5])
        s = claims.score(self.bank)
        self.assertEqual(s["by_source"]["good.example"], {"n": 1, "hits": 1, "hit_rate": 1.0, "p_vs_coin": 1.0})
        self.assertEqual(s["by_source"]["bad.example"]["hits"], 0)
        self.assertEqual(len(s["pending"]), 1)

    def test_future_claims_are_refused(self):
        with self.assertRaises(ValueError):
            claims.note(self.bank, "X", "t", "https://x.example", known_at="2999-01-01T00:00:00Z")
