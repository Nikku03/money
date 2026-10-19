from brain import ahead, changes
from brain.bank import Fact
from tests.util import TempHome

FOMC_PAGE = """
<div class="panel-heading"><h4><a id="1">2026 FOMC Meetings</a></h4></div>
<div class="row fomc-meeting">
<div class="fomc-meeting__month col-xs-5"><strong>October</strong></div>
<div class="fomc-meeting__date col-xs-4 col-lg-1">27-28</div></div>
<div class="row fomc-meeting">
<div class="fomc-meeting--shaded fomc-meeting__month col-xs-5"><strong>December</strong></div>
<div class="fomc-meeting__date col-xs-4 col-lg-1">8-9*</div></div>
<div class="panel-heading"><h4><a id="2">2027 FOMC Meetings</a></h4></div>
<div class="row fomc-meeting">
<div class="fomc-meeting__month col-xs-5"><strong>Jan/Feb</strong></div>
<div class="fomc-meeting__date col-xs-4 col-lg-1">31-1</div></div>
"""


def eight_k(t, known_at, items="2.02,9.01"):
    return Fact(entity=t, attr="filing", value="8-K", as_of=known_at[:10], known_at=known_at, tier="filed",
                source={"name": "SEC EDGAR", "url": f"https://www.sec.gov/{t}/{known_at}"},
                context={"accn": f"{t}-{known_at}", "items": items})


def filing(t, form, known_at):
    return Fact(entity=t, attr="filing", value=form, as_of=known_at[:10], known_at=known_at, tier="filed",
                source={"name": "SEC EDGAR", "url": f"https://www.sec.gov/{t}/{form}/{known_at}"},
                context={"accn": f"{t}-{form}-{known_at}"})


class ScheduleTest(TempHome):
    def test_fomc_calendar(self):
        got = ahead.parse_fomc(FOMC_PAGE)
        self.assertEqual(got, [
            {"kind": "fomc", "start": "2026-10-27", "date": "2026-10-28", "projections": False},
            {"kind": "fomc", "start": "2026-12-08", "date": "2026-12-09", "projections": True},
            {"kind": "fomc", "start": "2027-01-31", "date": "2027-02-01", "projections": False}])

    def test_market_calendar(self):
        got = ahead.market_calendar("2026-12-01", "2026-12-31")
        self.assertEqual(got, [{"kind": "expiry", "date": "2026-12-18", "what": "quarterly options and futures expiry"},
                               {"kind": "period_end", "date": "2026-12-31", "what": "quarter end"}])
        self.assertEqual(ahead.third_friday(2026, 10).isoformat(), "2026-10-16")

    def test_expected_earnings_from_the_company_history(self):
        # Releases on Tuesdays a year apart, after the close (21:05 UTC is 17:05 in New York).
        for k in ("2024-10-22T21:05:00Z", "2025-01-28T21:05:00Z", "2025-04-22T21:05:00Z", "2025-07-22T21:05:00Z",
                  "2025-10-21T21:05:00Z", "2026-01-27T21:05:00Z", "2026-04-21T21:05:00Z", "2026-07-21T21:05:00Z"):
            self.bank.add(eight_k("ACME", k))
        self.bank.add(eight_k("ACME", "2026-09-01T12:00:00Z", items="5.02"))   # not an earnings release
        view = self.bank.view("2026-10-19")
        got = ahead.expected_earnings(view, "ACME", "2026-10-19", "2026-10-26")
        self.assertEqual((got["date"], got["based_on"], got["shift_days"], got["spread_days"]),
                         ("2026-10-20", "2025-10-21", 0, 0))
        self.assertIsNone(ahead.expected_earnings(view, "ACME", "2026-11-01", "2026-11-08"))
        # Once this quarter's release is out, it is no longer expected.
        self.bank.add(eight_k("ACME", "2026-10-20T21:05:00Z"))
        self.assertIsNone(ahead.expected_earnings(self.bank.view("2026-10-21"), "ACME", "2026-10-19", "2026-10-26"))
        # A view before that release still expects it: nothing known later leaks in.
        self.assertIsNotNone(ahead.expected_earnings(self.bank.view("2026-10-19"), "ACME", "2026-10-19", "2026-10-26"))


class SurfaceTest(TempHome):
    def test_form_144_notices_in_the_last_week(self):
        self.bank.add_many([filing("ACME", "144", "2026-10-15T20:00:00Z"), filing("ACME", "144", "2026-10-01T20:00:00Z"),
                            filing("ACME", "4", "2026-10-16T20:00:00Z")])
        got = ahead.insider_sale_notices(self.bank.view("2026-10-19"), "ACME")
        self.assertEqual((got["n"], got["last"]), (1, "2026-10-15"))
        self.assertIsNone(ahead.insider_sale_notices(self.bank.view("2026-10-14"), "ACME"))

    def test_quiet_change_is_unusual_for_the_company(self):
        vals = [0.97, 0.975, 0.972, 0.968, 0.974, 0.90]
        for i, v in enumerate(vals):
            k = f"20{20 + i}-02-20T20:00:00Z"
            self.bank.add(Fact(entity="ACME", attr=changes.ATTR, value=v, as_of=k[:10], known_at=k, tier="derived",
                               source={"name": "t", "url": "https://x.test"}, inputs=("ACME/x",),
                               context={"form": "10-K", "method": changes.METHOD, "new_share": 0.4}))
        got = ahead.quiet_change(self.bank.view("2025-02-25"), "ACME")
        self.assertEqual((got["similarity"], got["usual"]), (0.90, 0.9718))
        z, f = changes.unusual(self.bank.view("2025-02-25"), "ACME")
        self.assertAlmostEqual(z, got["z"], places=2)
        self.assertLess(got["z"], -1.5)
        self.assertIsNone(ahead.quiet_change(self.bank.view("2024-02-25"), "ACME"))   # 0.974 is usual
        self.assertIsNone(ahead.quiet_change(self.bank.view("2025-04-25"), "ACME"))   # filed too long ago


class LookAheadTest(TempHome):
    def test_everything_together(self):
        for k in ("2024-10-22T21:05:00Z", "2025-10-21T21:05:00Z"):
            self.bank.add(eight_k("ACME", k))
        self.bank.add(filing("ACME", "144", "2026-10-15T20:00:00Z"))
        out = ahead.look_ahead(self.bank, ["ACME"], "2026-10-19", 10, fomc=ahead.parse_fomc(FOMC_PAGE),
                               at="2026-10-19")
        self.assertEqual([(x["kind"], x["date"]) for x in out["scheduled"]],
                         [("earnings", "2026-10-20"), ("fomc", "2026-10-28")])
        self.assertEqual([x["kind"] for x in out["surface"]], ["form144"])
        text = ahead.render(out)
        self.assertIn("earnings  ACME", text)
        self.assertIn("not a forecast", text)
