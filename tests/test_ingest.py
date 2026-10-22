import urllib.error
from unittest import mock

from brain import ingest
from brain.net import FetchError
from tests.util import TempHome


def _missing():
    err = urllib.error.HTTPError("https://data.sec.gov/api/xbrl/companyfacts/CIK0000884394.json", 404, "Not Found",
                                 None, None)
    try:
        raise FetchError("companyfacts: HTTP 404") from err
    except FetchError as e:
        return e


class IngestTest(TempHome):
    def test_a_fund_without_xbrl_statements_still_takes_filings_and_prices(self):
        with mock.patch.object(ingest.sec, "cik_for", return_value=884394), \
             mock.patch.object(ingest.sec, "facts_for", side_effect=_missing()), \
             mock.patch.object(ingest.sec, "filings_for", return_value=([], [])), \
             mock.patch.object(ingest.prices, "ingest", return_value=[]) as px:
            counts = ingest.ingest(self.bank, "spy", ("sec", "filings", "prices"), log=lambda m: None)
        self.assertEqual(counts, {"sec_facts": 0, "filings": 0, "identity": 0, "prices": 0})
        px.assert_called_once_with("SPY")

    def test_a_delisted_ticker_without_prices_does_not_stop_the_ingest(self):
        for gone in (_missing(), ValueError("no price rows for GONE")):
            with mock.patch.object(ingest.prices, "ingest", side_effect=gone):
                counts = ingest.ingest(self.bank, "gone", ("prices",), log=lambda m: None)
            self.assertEqual(counts, {"prices": 0})
        with mock.patch.object(ingest.prices, "ingest", side_effect=ValueError("bad body")), self.assertRaises(ValueError):
            ingest.ingest(self.bank, "odd", ("prices",), log=lambda m: None)

    def test_other_sec_failures_still_stop_the_ingest(self):
        with mock.patch.object(ingest.sec, "cik_for", return_value=1), \
             mock.patch.object(ingest.sec, "facts_for", side_effect=FetchError("gave up after 4 tries")), \
             self.assertRaises(FetchError):
            ingest.ingest(self.bank, "AAPL", ("sec",), log=lambda m: None)
