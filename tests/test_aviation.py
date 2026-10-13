import argparse
import contextlib
import io
import json
import zipfile
from unittest import mock


from brain import config
from brain.sources import aviation as av
from tests.util import TempHome, filed

MASTER_HEADER = ("N-NUMBER,SERIAL NUMBER,MFR MDL CODE,ENG MFR MDL,YEAR MFR,TYPE REGISTRANT,NAME,STREET,STREET2,CITY,"
                 "STATE,ZIP CODE,REGION,COUNTY,COUNTRY,LAST ACTION DATE,CERT ISSUE DATE,CERTIFICATION,TYPE AIRCRAFT,"
                 "TYPE ENGINE,STATUS CODE,MODE S CODE,FRACT OWNER,AIR WORTH DATE,OTHER NAMES(1),OTHER NAMES(2),"
                 "OTHER NAMES(3),OTHER NAMES(4),OTHER NAMES(5),EXPIRATION DATE,UNIQUE ID,KIT MFR, KIT MODEL,"
                 "MODE S CODE HEX,")
COLS = [c.strip() for c in MASTER_HEADER.split(",")]


def master_row(n, name, kind, hex_, status="V", model="1234567", year="2019", other=""):
    vals = {c: "" for c in COLS}
    vals.update({"N-NUMBER": f"{n}  ", "MFR MDL CODE": model, "YEAR MFR": year, "TYPE REGISTRANT": kind,
                 "NAME": f"{name}                ", "STREET": "1 SOME STREET", "CITY": "SOMEWHERE",
                 "STATUS CODE": f"{status} ", "OTHER NAMES(1)": other, "MODE S CODE HEX": f"{hex_.upper()}  "})
    return ",".join(vals[c] for c in COLS)


def registry_zip(rows) -> bytes:
    acft = "﻿CODE,MFR,MODEL,TYPE-ACFT,\r\n1234567,GULFSTREAM AEROSPACE   ,G-VII       ,5,\r\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("MASTER.txt", ("﻿" + MASTER_HEADER + "\r\n" + "\r\n".join(rows) + "\r\n").encode())
        z.writestr("ACFTREF.txt", acft.encode())
    return buf.getvalue()


ROWS = [
    master_row("100AC", "ACME CORP", "3", "a00001"),                      # the company: corporation
    master_row("200AC", "ACME CORPORATION", "7", "a00002", year="2005"),  # the company: LLC type, other suffix
    master_row("300AC", "ACME CORP", "1", "a00003"),                      # individual with the same name: never
    master_row("400AC", "ACME AVIATION LLC", "7", "a00004"),              # a different LLC: no expansion
    master_row("500AC", "ACME CORP", "4", "a00005"),                      # co-owned: never
    master_row("600AC", "ACME CORP", "3", "a00006", status="E"),          # revoked registration
    master_row("700AC", "ACME CORP", "3", "a00007", other="SMITH JOHN"),  # co-owner names listed
    master_row("800AC", "ACME CORP", "3", "a00008"),                      # opted out by tail
    master_row("900AC", "ACME CORP", "3", "a00009"),                      # opted out by hex
    master_row("1AC", "SOME TRUST CO TRUSTEE", "3", "a0000a"),
]

AIRPORTS = (b"id,ident,type,name,latitude_deg,longitude_deg\n"
            b"1,KSJC,large_airport,San Jose Intl,37.3626,-121.929\n"
            b"2,KTEB,medium_airport,Teterboro,40.85,-74.0608\n"
            b"3,XHEL,heliport,A heliport,40.70,-74.00\n"
            b"4,XCLO,closed,Closed field,37.00,-122.00\n")


class NamesTest(TempHome):
    def test_normalize(self):
        n = av.normalize_name
        self.assertEqual(n("COSTCO WHOLESALE CORP /NEW"), n("Costco Wholesale Corporation"))
        self.assertEqual(n("BANK OF AMERICA CORP /DE/"), "BANKOFAMERICA")
        self.assertEqual(n("WELLS FARGO & COMPANY/MN"), n("WELLS FARGO AND CO"))
        self.assertEqual(n("ELI LILLY & Co"), n("ELI LILLY AND COMPANY"))
        self.assertEqual(n("AMAZON COM INC"), n("AMAZON.COM, INC."))
        self.assertEqual(n("Seagate Technology Holdings plc"), "SEAGATETECHNOLOGY")
        self.assertEqual(n("The Walt Disney Co"), n("WALT DISNEY CO"))
        self.assertNotEqual(n("AMAZON COM INC"), n("AMAZON.COM SERVICES LLC"))
        self.assertEqual(n("CO"), "CO")  # never empties a name

    def test_reg(self):
        self.assertEqual(av.norm_reg("123ab"), "N123AB")
        self.assertEqual(av.norm_reg("N5"), "N5")


class RegistryTest(TempHome):
    def setUp(self):
        super().setUp()
        (config.home() / "aviation_optout.txt").write_text("# never these\nN800AC\na00009  # by hex\n")
        self.index = av.parse_registry(registry_zip(ROWS))

    def test_only_the_company_itself(self):
        hits = av.company_aircraft("Acme Corp.", self.index)
        self.assertEqual([h["reg"] for h in hits], ["N100AC", "N200AC"])
        self.assertEqual(hits[0]["hex"], "a00001")
        self.assertEqual(hits[0]["model"], "GULFSTREAM AEROSPACE G-VII")
        self.assertEqual(hits[1]["year"], 2005)
        self.assertEqual({h["registrant"] for h in hits}, {"corporation", "LLC"})

    def test_individuals_never_in_the_index(self):
        everything = [a for lst in self.index.values() for a in lst]
        self.assertNotIn("N300AC", [a["reg"] for a in everything])   # type 1, individual
        self.assertNotIn("N500AC", [a["reg"] for a in everything])   # type 4, co-owned
        self.assertNotIn("N700AC", [a["reg"] for a in everything])   # co-owner names
        self.assertNotIn("N600AC", [a["reg"] for a in everything])   # not a valid registration
        for a in everything:  # no addresses kept
            self.assertEqual(set(a), {"reg", "hex", "model", "year", "owner", "registrant"})

    def test_no_expansion_to_other_llcs(self):
        self.assertEqual(av.company_aircraft("ACME AVIATION", self.index)[0]["reg"], "N400AC")
        self.assertNotIn("N400AC", [h["reg"] for h in av.company_aircraft("ACME CORP", self.index)])

    def test_optout(self):
        regs = [h["reg"] for h in av.company_aircraft("ACME CORP", self.index)]
        self.assertNotIn("N800AC", regs)
        self.assertNotIn("N900AC", regs)
        self.assertTrue(av.opted_out("800AC", None))
        self.assertTrue(av.opted_out(None, "A00009"))
        self.assertFalse(av.opted_out("N100AC", "a00001"))

    def test_link_writes_facts_and_the_watched_set(self):
        self.bank.add(filed("ACME", "identity.name", "ACME CORP /DE/", "2026-10-01", "2026-10-01T00:00:00Z"))
        n = av.link(self.bank, "acme", self.index, "2026-10-10T12:00:00Z")
        self.assertEqual(n, 2)
        fs = av.registered(self.bank.view(), "ACME")
        self.assertEqual([f.value for f in fs], ["N100AC", "N200AC"])
        f = fs[0]
        self.assertEqual((f.attr, f.tier, f.as_of, f.known_at), ("aircraft.registered", "reported", "2026-10-10",
                                                                  "2026-10-10T12:00:00Z"))
        self.assertEqual(set(f.context), {"hex", "model", "year", "owner"})
        self.assertEqual(f.source["url"], av.REGISTRY_URL)
        # Re-linking against the same download adds nothing.
        before = len(self.bank.facts("ACME", "aircraft.registered"))
        av.link(self.bank, "ACME", self.index, "2026-10-10T12:00:00Z")
        self.assertEqual(len(self.bank.facts("ACME", "aircraft.registered")), before)
        links = json.loads((config.home() / "aviation_links.json").read_text())
        self.assertEqual([a["reg"] for a in links["ACME"]["aircraft"]], ["N100AC", "N200AC"])
        # A later download without N200AC: only N100AC is current.
        later = av.parse_registry(registry_zip([ROWS[0]]))
        av.link(self.bank, "ACME", later, "2026-10-12T12:00:00Z")
        self.assertEqual([f.value for f in av.registered(self.bank.view(), "ACME")], ["N100AC"])

    def test_link_needs_identity(self):
        with self.assertRaises(KeyError):
            av.link(self.bank, "NOPE", self.index, "2026-10-10T12:00:00Z")


T0 = 1_791_000_000.0  # 2026-10-03, in the past so a View of now sees what poll writes


def obs(t, ground=False, alt=10000.0, lat=37.3626, lon=-121.929, flags=0):
    return {"t": t, "ground": ground, "alt": 0 if ground else alt, "lat": lat, "lon": lon, "flags": flags}


class LandingTest(TempHome):
    def setUp(self):
        super().setUp()
        self.apts = av.parse_airports(AIRPORTS)

    def test_airport_types_and_radius(self):
        self.assertEqual(len(self.apts["ident"]), 2)  # heliport and closed field dropped
        self.assertEqual(av.nearest_airport(self.apts, 37.37, -121.93)["ident"], "KSJC")
        self.assertIsNone(av.nearest_airport(self.apts, 37.50, -121.93))  # ~15 km away

    def test_ground_air_ground(self):
        st, ev = av.step(None, obs(T0, ground=True), T0, self.apts)
        self.assertEqual((st["phase"], st["airport"], ev), ("ground", "KSJC", None))
        st, ev = av.step(st, obs(T0 + 900, alt=8000), T0 + 900, self.apts)
        self.assertEqual(ev["kind"], "takeoff")
        self.assertEqual(st["from"], "KSJC")
        st, ev = av.step(st, obs(T0 + 3600, alt=41000, lat=39.0, lon=-100.0), T0 + 3600, self.apts)
        self.assertIsNone(ev)
        st, ev = av.step(st, obs(T0 + 5 * 3600, ground=True, lat=40.851, lon=-74.06), T0 + 5 * 3600, self.apts)
        self.assertEqual(ev["kind"], "landing")
        self.assertEqual((ev["from"], ev["to"], ev["to_name"]), ("KSJC", "KTEB", "Teterboro"))
        self.assertEqual(ev["departed"], av._iso(T0 + 900))
        self.assertEqual(ev["arrived"], av._iso(T0 + 5 * 3600))
        self.assertEqual(st["airport"], "KTEB")

    def test_low_then_gone_is_a_landing(self):
        st = {"phase": "air", "from": "KSJC", "from_name": "San Jose Intl", "departed": av._iso(T0)}
        st, _ = av.step(st, obs(T0 + 3000, alt=1200, lat=40.86, lon=-74.07), T0 + 3000, self.apts)
        st2, ev = av.step(st, None, T0 + 3000 + 600, self.apts)  # gone 10 min: not yet
        self.assertIsNone(ev)
        self.assertEqual(st2["phase"], "air")
        st3, ev = av.step(st2, None, T0 + 3000 + 1500, self.apts)  # gone 25 min
        self.assertEqual((ev["kind"], ev["to"], ev["arrived"]), ("landing", "KTEB", av._iso(T0 + 3000)))
        self.assertEqual(st3["phase"], "ground")

    def test_high_then_gone_is_not_a_landing(self):
        st = {"phase": "air", "from": "KSJC", "from_name": "x", "departed": av._iso(T0)}
        st, _ = av.step(st, obs(T0, alt=35000, lat=40.86, lon=-74.07), T0, self.apts)
        st, ev = av.step(st, None, T0 + 3600, self.apts)
        self.assertIsNone(ev)
        st, ev = av.step(st, None, T0 + 13 * 3600, self.apts)
        self.assertEqual((st["phase"], ev), ("unknown", None))

    def test_low_and_gone_away_from_airports_records_nothing(self):
        st = {"phase": "air", "from": None, "from_name": None, "departed": None}
        st, _ = av.step(st, obs(T0, alt=900, lat=38.5, lon=-120.0), T0, self.apts)
        st, ev = av.step(st, None, T0 + 1800, self.apts)
        self.assertEqual((st["phase"], ev), ("unknown", None))

    def test_observe(self):
        o = av.observe({"hex": "a00001", "alt_baro": "ground", "lat": 1.0, "lon": 2.0, "seen": 3}, 100.0)
        self.assertEqual((o["ground"], o["alt"], o["t"]), (True, 0, 97.0))
        o = av.observe({"hex": "a00001", "alt_baro": 2500, "lat": 1.0, "lon": 2.0, "dbFlags": 8}, 100.0)
        self.assertEqual((o["ground"], o["alt"], o["flags"]), (False, 2500.0, 8))
        self.assertIsNone(av.observe({"hex": "a00001"}, 100.0)["ground"])


class PollTest(TempHome):
    def setUp(self):
        super().setUp()
        self.bank.add(filed("ACME", "identity.name", "ACME CORP", "2026-10-01", "2026-10-01T00:00:00Z"))
        self.index = av.parse_registry(registry_zip(ROWS + [master_row("42AC", "ACME CORP", "3", "a00042")]))
        (config.home() / "aviation_optout.txt").write_text("N800AC\na00009\n")
        av.link(self.bank, "ACME", self.index, "2026-10-10T12:00:00Z")
        self.apts = av.parse_airports(AIRPORTS)
        self.calls = []
        self.answers = {}

    def get(self, hex_):
        self.calls.append(hex_)
        return {"ac": self.answers.get(hex_, [])}

    def poll(self, now):
        return av.poll(self.bank, tickers=["ACME"], get=self.get, table=self.apts, now=now, log=lambda *_: None)

    def test_landing_written_once_with_the_spec_shape(self):
        self.answers = {"a00001": [{"hex": "a00001", "alt_baro": "ground", "lat": 37.3626, "lon": -121.929}]}
        self.assertEqual(self.poll(T0), 0)
        self.answers = {"a00001": [{"hex": "a00001", "alt_baro": 9000, "lat": 37.5, "lon": -121.5}]}
        self.assertEqual(self.poll(T0 + 900), 0)
        self.answers = {"a00001": [{"hex": "a00001", "alt_baro": "ground", "lat": 40.85, "lon": -74.06}]}
        self.assertEqual(self.poll(T0 + 6 * 3600), 1)
        f = self.bank.facts("ACME", "aircraft.flight")[0]
        self.assertEqual((f.value, f.tier, f.as_of), ("KTEB", "reported", av._iso(T0 + 6 * 3600)[:10]))
        self.assertEqual(set(f.context), {"hex", "reg", "from", "to", "from_name", "to_name", "departed", "arrived"})
        self.assertEqual((f.context["from"], f.context["reg"]), ("KSJC", "N100AC"))
        self.assertEqual(f.source["name"], av.ADSB_SOURCE["name"])
        self.assertEqual(av.flights(self.bank.view(), "ACME")[0].id, f.id)
        state = json.loads((config.home() / "aviation_state.json").read_text())
        for st in state["aircraft"].values():  # only the last observation is kept, no track
            self.assertLessEqual(set(st), {"phase", "t", "alt", "lat", "lon", "airport", "airport_name", "from",
                                           "from_name", "departed"})

    def test_opted_out_and_unregistered_never_polled(self):
        self.poll(T0)
        self.assertEqual(sorted(self.calls), ["a00001", "a00002", "a00042"])  # not a00003..a00009
        (config.home() / "aviation_optout.txt").write_text("N42AC\n")
        self.calls = []
        self.poll(T0 + 60)
        self.assertNotIn("a00042", self.calls)

    def test_ladd_flag_drops_the_aircraft_for_good(self):
        self.answers = {"a00002": [{"hex": "a00002", "alt_baro": "ground", "lat": 37.3626, "lon": -121.929,
                                    "dbFlags": 8}]}
        self.poll(T0)
        state = json.loads((config.home() / "aviation_state.json").read_text())
        self.assertIn("a00002", state["blocked"])
        self.assertNotIn("a00002", state["aircraft"])
        self.calls = []
        self.poll(T0 + 60)
        self.assertNotIn("a00002", self.calls)

    def test_stale_link_is_rechecked_and_sold_aircraft_dropped(self):
        links_p = config.home() / "aviation_links.json"
        links = json.loads(links_p.read_text())
        links["ACME"]["linked_at"] = "2026-09-01T00:00:00Z"  # older than a week
        links_p.write_text(json.dumps(links))
        sold = av.parse_registry(registry_zip([ROWS[0]]))  # only N100AC is still the company's
        with mock.patch.object(av, "registry", return_value=(sold, "2026-10-12T00:00:00Z")):
            self.poll(T0)
        self.assertEqual(self.calls, ["a00001"])

    def test_no_companies(self):
        self.assertEqual(av.poll(self.bank, tickers=[], get=self.get, table=self.apts, log=lambda *_: None), 0)
        self.assertEqual(self.calls, [])


class CliTest(TempHome):
    def test_subcommands(self):
        p = argparse.ArgumentParser()
        av.add_cli(p.add_subparsers(dest="cmd"))
        a = p.parse_args(["jets", "link", "AAPL", "WMT"])
        self.assertEqual((a.tickers, a.fn), (["AAPL", "WMT"], av._cmd_link))
        self.assertEqual(p.parse_args(["jets", "poll"]).fn, av._cmd_poll)
        a = p.parse_args(["jets", "optout", "N1AB"])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(a.fn(a), 0)
        self.assertIn("N1AB", av.optout())


if __name__ == "__main__":
    import unittest
    unittest.main()
