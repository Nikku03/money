import argparse
import contextlib
import io
import json
import math
import threading
import time
import zipfile
from unittest import mock


from brain import config
from brain.bank import Bank, Fact
from brain.net import FetchError
from brain.sources import aviation as av
from brain.timeutil import now_iso
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
    master_row("200AC", "ACME CORPORATION", "3", "a00002", year="2005"),  # the company, its form spelled out
    master_row("300AC", "ACME CORP", "1", "a00003"),                      # individual with the same name: never
    master_row("400AC", "ACME AVIATION LLC", "7", "a00004"),              # a different LLC: no expansion
    master_row("500AC", "ACME CORP", "4", "a00005"),                      # co-owned: never
    master_row("600AC", "ACME CORP", "3", "a00006", status="E"),          # revoked registration
    master_row("700AC", "ACME CORP", "3", "a00007", other="SMITH JOHN"),  # co-owner names listed
    master_row("800AC", "ACME CORP", "3", "a00008"),                      # opted out by tail
    master_row("900AC", "ACME CORP", "3", "a00009"),                      # opted out by hex
    master_row("1AC", "SOME TRUST CO TRUSTEE", "3", "a0000a"),
    master_row("210AC", "ACME CORP", "7", "a00010"),                      # an LLC registrant: not Acme Corp
    master_row("220AC", "ACME HOLDINGS LLC", "7", "a00011"),              # another company, whatever HOLDINGS means
    master_row("230AC", "ACME LLC", "7", "a00012"),                       # an LLC of the same core name: never
    master_row("240AC", "ACME INC", "3", "a00013"),                       # another legal form: another company
]

# Registrants in the real FAA registry that a looser rule linked to universe companies (WLK, APA, GAP).
COLLISIONS = [
    master_row("479JS", "WEST LAKE HOLDINGS INC", "3", "b00001"),
    master_row("58CV", "APA LLC", "7", "b00002"),
    master_row("3704G", "GAP HOLDINGS LLC", "7", "b00003"),
    master_row("822SF", "GAP INC", "3", "b00004"),
    master_row("5WL", "WESTLAKE CORPORATION", "3", "b00005"),
]

AIRPORTS = (b"id,ident,type,name,latitude_deg,longitude_deg,elevation_ft\n"
            b"1,KSJC,large_airport,San Jose Intl,37.362452,-121.929188,62\n"
            b"2,KTEB,medium_airport,Teterboro,40.85,-74.0608,9\n"
            b"3,XHEL,heliport,A heliport,40.70,-74.00,10\n"
            b"4,XCLO,closed,Closed field,37.00,-122.00,10\n"
            b"5,KRHV,small_airport,Reid-Hillview,37.332901,-121.819,135\n"
            b"6,KASE,medium_airport,Aspen-Pitkin County,39.223202,-106.869003,7820\n"
            b"7,CO03,small_airport,Aspen Gliderport,39.2599983215332,-106.91200256347656,8440\n"
            b"8,KAPA,medium_airport,Centennial,39.57009888,-104.848999,5885\n"
            b"9,1CO8,small_airport,Everitt,39.52920150756836,-104.65799713134766,6295\n"
            b"10,KLGA,large_airport,LaGuardia,40.777199,-73.872597,21\n")


def pt(lat, lon, km, brg):
    """The point ``km`` from (lat, lon) on bearing ``brg`` degrees."""
    d, la, lo, b = km / 6371.0, math.radians(lat), math.radians(lon), math.radians(brg)
    la2 = math.asin(math.sin(la) * math.cos(d) + math.cos(la) * math.sin(d) * math.cos(b))
    lo2 = lo + math.atan2(math.sin(b) * math.sin(d) * math.cos(la), math.cos(d) - math.sin(la) * math.sin(la2))
    return math.degrees(la2), math.degrees(lo2)


class NamesTest(TempHome):
    def test_normalize(self):
        n = av.normalize_name
        self.assertEqual(n("COSTCO WHOLESALE CORP /NEW"), n("Costco Wholesale Corporation"))
        self.assertEqual(n("BANK OF AMERICA CORP /DE/"), "BANK OF AMERICA CORP")
        self.assertEqual(n("WELLS FARGO & COMPANY/MN"), n("WELLS FARGO AND CO"))
        self.assertEqual(n("ELI LILLY & Co"), n("ELI LILLY AND COMPANY"))
        self.assertEqual(n("AMAZON COM INC"), n("AMAZON.COM, INC."))
        self.assertEqual(n("Seagate Technology Holdings plc"), "SEAGATE TECHNOLOGY HOLDINGS PLC")
        self.assertEqual(n("The Walt Disney Co"), n("WALT DISNEY CO"))
        self.assertEqual(n("Acme L.L.C."), n("ACME LLC"))
        self.assertNotEqual(n("AMAZON COM INC"), n("AMAZON.COM SERVICES LLC"))
        self.assertEqual(n("CO"), "CO")  # never empties a name

    def test_words_are_never_dropped_or_joined(self):
        n = av.normalize_name
        self.assertNotEqual(n("WESTLAKE CORP"), n("WEST LAKE HOLDINGS INC"))  # joined words, HOLDINGS dropped
        self.assertNotEqual(n("WESTLAKE CORP"), n("WEST LAKE CORP"))
        self.assertNotEqual(n("GAP INC"), n("GAP HOLDINGS LLC"))
        self.assertNotEqual(n("GAP INC"), n("GAP HOLDINGS INC"))
        self.assertNotEqual(n("APA Corp"), n("APA LLC"))
        self.assertNotEqual(n("ACME CORP"), n("ACME INC"))
        self.assertTrue(av.is_llc(n("Acme, L.L.C.")))
        self.assertFalse(av.is_llc(n("ACME CORP")))

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
        self.assertEqual({h["registrant"] for h in hits}, {"corporation"})

    def test_individuals_never_in_the_index(self):
        everything = [a for lst in self.index.values() for a in lst]
        self.assertNotIn("N300AC", [a["reg"] for a in everything])   # type 1, individual
        self.assertNotIn("N500AC", [a["reg"] for a in everything])   # type 4, co-owned
        self.assertNotIn("N700AC", [a["reg"] for a in everything])   # co-owner names
        self.assertNotIn("N600AC", [a["reg"] for a in everything])   # not a valid registration
        for a in everything:  # no addresses kept
            self.assertEqual(set(a), {"reg", "hex", "model", "year", "owner", "registrant"})

    def test_no_expansion_to_other_llcs(self):
        self.assertEqual([h["reg"] for h in av.company_aircraft("ACME AVIATION LLC", self.index)], ["N400AC"])
        self.assertEqual(av.company_aircraft("ACME AVIATION", self.index), [])  # an LLC only for an LLC
        acme = [h["reg"] for h in av.company_aircraft("ACME CORP", self.index)]
        for other in ("N400AC", "N210AC", "N220AC", "N230AC", "N240AC"):
            self.assertNotIn(other, acme)
        self.assertEqual([h["reg"] for h in av.company_aircraft("Acme LLC", self.index)], ["N230AC"])

    def test_real_registry_collisions_never_match(self):
        index = av.parse_registry(registry_zip(COLLISIONS))
        regs = lambda name: [h["reg"] for h in av.company_aircraft(name, index)]
        self.assertEqual(regs("WESTLAKE CORP"), ["N5WL"])  # not WEST LAKE HOLDINGS INC
        self.assertEqual(regs("APA Corp"), [])              # not APA LLC
        self.assertEqual(regs("GAP INC"), ["N822SF"])       # not GAP HOLDINGS LLC

    def test_optout(self):
        regs = [h["reg"] for h in av.company_aircraft("ACME CORP", self.index)]
        self.assertNotIn("N800AC", regs)
        self.assertNotIn("N900AC", regs)
        self.assertTrue(av.opted_out("800AC", None))
        self.assertTrue(av.opted_out(None, "A00009"))
        self.assertFalse(av.opted_out("N100AC", "a00001"))

    def test_ladd_blocked_never_linked(self):
        (config.home() / "aviation_state.json").write_text(json.dumps({"aircraft": {}, "blocked": {"a00002": "LADD"}}))
        self.assertEqual([h["reg"] for h in av.company_aircraft("ACME CORP", self.index)], ["N100AC"])

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
        self.assertEqual(links["ACME"]["rule"], av.MATCH_RULE)
        # A later download without N200AC: only N100AC is current.
        later = av.parse_registry(registry_zip([ROWS[0]]))
        av.link(self.bank, "ACME", later, "2026-10-12T12:00:00Z")
        self.assertEqual([f.value for f in av.registered(self.bank.view(), "ACME")], ["N100AC"])

    def test_link_needs_identity(self):
        with self.assertRaises(KeyError):
            av.link(self.bank, "NOPE", self.index, "2026-10-10T12:00:00Z")

    def test_concurrent_links_keep_each_other(self):
        self.bank.add(filed("ACME", "identity.name", "ACME CORP", "2026-10-01", "2026-10-01T00:00:00Z"))
        self.bank.add(filed("BETA", "identity.name", "ACME LLC", "2026-10-01", "2026-10-01T00:00:00Z"))
        other = threading.Thread(target=lambda: av.link(Bank(), "BETA", self.index, "2026-10-10T12:00:00Z"))
        write = av._write_json

        def slow_write(p, obj):  # the other link runs while this one is between reading and writing
            if p.name == "aviation_links.json" and not other.is_alive() and other.ident is None:
                other.start()
                other.join(0.5)
            write(p, obj)
        with mock.patch.object(av, "_write_json", slow_write):
            av.link(self.bank, "ACME", self.index, "2026-10-10T12:00:00Z")
        other.join(10)
        links = json.loads((config.home() / "aviation_links.json").read_text())
        self.assertEqual(sorted(links), ["ACME", "BETA"])


T0 = 1_791_000_000.0  # 2026-10-03, in the past so a View of now sees what poll writes


def obs(t, ground=False, alt=10000.0, lat=37.3626, lon=-121.929, flags=0, track=None, rate=None):
    return {"t": t, "ground": ground, "alt": 0 if ground else alt, "lat": lat, "lon": lon, "track": track,
            "rate": rate, "flags": flags}


class LandingTest(TempHome):
    def setUp(self):
        super().setUp()
        self.apts = av.parse_airports(AIRPORTS)

    def step(self, st, o, now=None):
        return av.step(st, o, o["t"] if o is not None and now is None else now, self.apts)

    def test_airport_types_and_radius(self):
        self.assertEqual(len(self.apts["ident"]), 8)  # heliport and closed field dropped
        self.assertEqual(self.apts["elev"][self.apts["ident"].index("KASE")], 7820)
        self.assertEqual(list(self.apts["major"][:2]), [True, True])
        self.assertFalse(self.apts["major"][self.apts["ident"].index("KRHV")])
        self.assertEqual(av.nearest_airport(self.apts, 37.37, -121.93)["ident"], "KSJC")
        self.assertIsNone(av.nearest_airport(self.apts, 37.50, -121.93))  # ~15 km away

    def test_ground_air_ground(self):
        st, ev = self.step(None, obs(T0, ground=True))
        self.assertEqual((st["phase"], st["airport"], ev), ("ground", "KSJC", None))
        st, ev = self.step(st, obs(T0 + 900, alt=8000))
        self.assertEqual(ev["kind"], "takeoff")
        self.assertEqual(st["from"], "KSJC")
        for k in range(2, 20):  # seen at cruise every 15 minutes on the way
            st, ev = self.step(st, obs(T0 + k * 900, alt=41000, lat=39.0, lon=-100.0 + k))
            self.assertIsNone(ev)
        st, ev = self.step(st, obs(T0 + 5 * 3600, ground=True, lat=40.851, lon=-74.06))
        self.assertEqual(ev["kind"], "landing")
        self.assertEqual((ev["from"], ev["to"], ev["to_name"]), ("KSJC", "KTEB", "Teterboro"))
        self.assertEqual(ev["departed"], av._iso(T0 + 900))
        self.assertEqual(ev["arrived"], av._iso(T0 + 5 * 3600))
        self.assertEqual(st["airport"], "KTEB")

    def test_short_coverage_gap_keeps_the_leg(self):
        st = {"phase": "air", "from": "KSJC", "from_name": "San Jose Intl", "departed": av._iso(T0)}
        st, _ = self.step(st, obs(T0 + 3600, alt=41000, lat=39.0, lon=-100.0))
        st, ev = self.step(st, obs(T0 + 3600 + 1800, alt=41000, lat=39.5, lon=-95.0))  # one poll missed
        self.assertEqual((ev, st["from"]), (None, "KSJC"))

    def test_unseen_stop_starts_a_new_leg(self):
        st, _ = self.step(None, obs(T0, ground=True, lat=40.85, lon=-74.0608))  # KTEB
        st, ev = self.step(st, obs(T0 + 900, alt=35000.0, lat=40.0, lon=-80.0))
        self.assertEqual(ev["from"], "KTEB")
        st, ev = self.step(st, None, T0 + 3 * 3600)  # landed somewhere unseen
        st, ev = self.step(st, obs(T0 + 5 * 3600, alt=30000.0, lat=36.0, lon=-110.0))  # the next leg
        self.assertEqual((ev, st["from"], st["departed"]), (None, None, None))
        st, ev = self.step(st, obs(T0 + 7 * 3600, ground=True))
        self.assertEqual((ev["from"], ev["departed"], ev["to"]), (None, None, "KSJC"))  # not KTEB -> KSJC

    def test_seen_on_the_ground_after_a_long_gap_has_no_known_origin(self):
        st = {"phase": "air", "from": "KTEB", "from_name": "Teterboro", "departed": av._iso(T0)}
        st, _ = self.step(st, obs(T0 + 900, alt=35000.0, lat=40.0, lon=-80.0))
        st, ev = self.step(st, obs(T0 + 6 * 3600, ground=True))
        self.assertEqual((ev["from"], ev["to"]), (None, "KSJC"))

    def test_ground_state_does_not_outlive_an_unseen_hop(self):
        parked, _ = self.step(None, obs(T0, ground=True, lat=40.85, lon=-74.0608))  # KTEB
        days = T0 + 3 * 86400
        st, ev = self.step(parked, obs(days, alt=3000, lat=37.40, lon=-121.95))  # climbing out of San Jose
        self.assertEqual((ev["kind"], ev["from"], st["from"]), ("takeoff", None, None))
        lat, lon = pt(40.85, -74.0608, 10, 45)  # 10 km out of Teterboro, days later: left from there
        st, ev = self.step(parked, obs(days, alt=3000, lat=lat, lon=lon))
        self.assertEqual((ev["from"], ev["departed"]), ("KTEB", av._iso(days)))

    def test_low_then_gone_is_a_landing(self):
        st = {"phase": "air", "from": "KSJC", "from_name": "San Jose Intl", "departed": av._iso(T0)}
        st, _ = self.step(st, obs(T0 + 3000, alt=1200, lat=40.86, lon=-74.07))
        st2, ev = self.step(st, None, T0 + 3000 + 600)  # gone 10 min: not yet
        self.assertIsNone(ev)
        self.assertEqual(st2["phase"], "air")
        st3, ev = self.step(st2, None, T0 + 3000 + 1500)  # gone 25 min
        self.assertEqual((ev["kind"], ev["to"], ev["arrived"]), ("landing", "KTEB", av._iso(T0 + 3000)))
        self.assertEqual(st3["phase"], "ground")

    def above(self, ident, alt):
        """Last seen 1.1 km north of a field at ``alt`` ft (pressure altitude), then gone 25 minutes."""
        i = self.apts["ident"].index(ident)
        lat, lon = math.degrees(self.apts["lat"][i]) + 0.01, math.degrees(self.apts["lon"][i])
        st = {"phase": "air", "from": "KTEB", "from_name": "Teterboro", "departed": av._iso(T0)}
        st, _ = self.step(st, obs(T0 + 18000, alt=alt, lat=lat, lon=lon))
        return self.step(st, None, T0 + 18000 + 1500)

    def test_height_is_measured_above_the_field(self):
        st, ev = self.above("KASE", 9000.0)  # 1,180 ft above Aspen (7,820 ft), a gliderport 5 km off
        self.assertEqual((ev["to"], st["phase"]), ("KASE", "ground"))
        st, ev = self.above("KASE", 12000.0)  # 4,180 ft above it
        self.assertEqual((ev, st["phase"]), (None, "air"))

    def test_a_far_private_strip_is_never_guessed(self):
        st, ev = self.above("KAPA", 9000.0)  # 3,115 ft above Centennial; Everitt (private) 17 km off at 2.7 deg
        self.assertEqual((ev, st["phase"]), (None, "air"))

    def test_approach_past_a_small_strip_lands_at_the_big_airport(self):
        for km in (6, 8, 10):  # on final to San Jose runway 30, from the south-east, Reid-Hillview nearby
            lat, lon = pt(37.362452, -121.929188, km, 135)
            alt = 62 + km * 1000 * math.tan(math.radians(3)) / 0.3048
            st = {"phase": "air", "from": "KTEB", "from_name": "Teterboro", "departed": av._iso(T0)}
            for track in (315.0, None):
                if km == 6 and track is None:
                    continue  # equally on a 3 degree path to both: only the track tells them apart
                last, _ = self.step(st, obs(T0 + 18000, alt=alt, lat=lat, lon=lon, track=track, rate=-700))
                _, ev = self.step(last, None, T0 + 18000 + 1500)
                self.assertEqual(ev and ev["to"], "KSJC", (km, track))

    def test_climb_out_then_gone_is_not_a_landing(self):
        lat, lon = pt(40.85, -74.0608, 3, 0)  # 3 km north of Teterboro, 2,500 ft; LaGuardia 3 degrees below
        for track, rate in ((0.0, 1800.0), (None, None)):
            st, _ = self.step(None, obs(T0, ground=True, lat=40.85, lon=-74.0608))
            st, ev = self.step(st, obs(T0 + 900, alt=2500.0, lat=lat, lon=lon, track=track, rate=rate))
            self.assertEqual(ev["kind"], "takeoff")
            st, ev = self.step(st, None, T0 + 900 + 1500)
            self.assertIsNone(ev, (track, rate))

    def test_high_then_gone_is_not_a_landing(self):
        st = {"phase": "air", "from": "KSJC", "from_name": "x", "departed": av._iso(T0)}
        st, _ = self.step(st, obs(T0, alt=35000, lat=40.86, lon=-74.07))
        st, ev = self.step(st, None, T0 + 3600)
        self.assertIsNone(ev)
        st, ev = self.step(st, None, T0 + 13 * 3600)
        self.assertEqual((st["phase"], ev), ("unknown", None))

    def test_low_and_gone_away_from_airports_records_nothing(self):
        st = {"phase": "air", "from": None, "from_name": None, "departed": None}
        st, _ = self.step(st, obs(T0, alt=900, lat=38.5, lon=-120.0))
        st, ev = self.step(st, None, T0 + 1800)
        self.assertEqual((st["phase"], ev), ("unknown", None))

    def test_observe(self):
        o = av.observe({"hex": "a00001", "alt_baro": "ground", "lat": 1.0, "lon": 2.0, "seen": 3}, 100.0)
        self.assertEqual((o["ground"], o["alt"], o["t"]), (True, 0, 97.0))
        o = av.observe({"hex": "a00001", "alt_baro": 2500, "lat": 1.0, "lon": 2.0, "dbFlags": 8, "track": 271.5,
                        "baro_rate": -704}, 100.0)
        self.assertEqual((o["ground"], o["alt"], o["flags"], o["track"], o["rate"]), (False, 2500.0, 8, 271.5, -704.0))
        self.assertIsNone(av.observe({"hex": "a00001"}, 100.0)["ground"])
        self.assertIsNone(av.observe({"hex": "a00001", "alt_baro": 900}, 100.0)["track"])


class PollBase(TempHome):
    def setUp(self):
        super().setUp()
        self.bank.add(filed("ACME", "identity.name", "ACME CORP", "2026-10-01", "2026-10-01T00:00:00Z"))
        self.index = av.parse_registry(registry_zip(ROWS + [master_row("42AC", "ACME CORP", "3", "a00042")]))
        (config.home() / "aviation_optout.txt").write_text("N800AC\na00009\n")
        av.link(self.bank, "ACME", self.index, now_iso())
        self.apts = av.parse_airports(AIRPORTS)
        self.calls = []
        self.answers = {}

    def get(self, hex_):
        self.calls.append(hex_)
        return {"ac": self.answers.get(hex_, [])}

    def poll(self, now):
        return av.poll(self.bank, tickers=["ACME"], get=self.get, table=self.apts, now=now, log=lambda *_: None)

    def set_link(self, **kw):
        p = config.home() / "aviation_links.json"
        links = json.loads(p.read_text())
        links["ACME"].update(kw)
        for k in [k for k, v in kw.items() if v is None]:
            links["ACME"].pop(k)
        p.write_text(json.dumps(links))


class PollTest(PollBase):
    def test_landing_written_once_with_the_spec_shape(self):
        self.answers = {"a00001": [{"hex": "a00001", "alt_baro": "ground", "lat": 37.3626, "lon": -121.929}]}
        self.assertEqual(self.poll(T0), 0)
        for k in range(1, 24):  # airborne, seen every 15 minutes
            self.answers = {"a00001": [{"hex": "a00001", "alt_baro": 9000 if k == 1 else 41000, "lat": 38.0,
                                        "lon": -121.5 + 2 * k, "track": 80.0}]}
            self.assertEqual(self.poll(T0 + k * 900), 0)
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
            self.assertLessEqual(set(st), {"phase", "t", "alt", "lat", "lon", "track", "rate", "airport",
                                           "airport_name", "from", "from_name", "departed"})

    def test_opted_out_and_unregistered_never_polled(self):
        self.poll(T0)
        self.assertEqual(sorted(self.calls), ["a00001", "a00002", "a00042"])  # not a00003..a00013
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
        self.assertNotIn("N200AC", [f.value for f in av.registered(self.bank.view(), "ACME")])

    def test_concurrent_polls_keep_each_others_ladd_blocks(self):
        ladd = {"a00002": [{"hex": "a00002", "alt_baro": "ground", "lat": 37.3626, "lon": -121.929, "dbFlags": 8}]}
        other = threading.Thread(target=lambda: av.poll(Bank(), tickers=["ACME"], get=lambda h: {"ac": ladd.get(h, [])},
                                                        table=self.apts, now=T0, log=lambda *_: None))

        def get(hex_):  # while this poll waits on adsb.lol, another process polls and sees the LADD flag
            if other.ident is None:
                other.start()
                other.join(0.5)
            return self.get(hex_)
        av.poll(self.bank, tickers=["ACME"], get=get, table=self.apts, now=T0, log=lambda *_: None)
        other.join(10)
        self.assertFalse(other.is_alive())
        state = json.loads((config.home() / "aviation_state.json").read_text())
        self.assertIn("a00002", state["blocked"])
        self.calls = []
        self.poll(T0 + 60)
        self.assertNotIn("a00002", self.calls)

    def test_stale_link_is_rechecked_and_sold_aircraft_dropped(self):
        # Re-linked a day ago, but against a registry download 8 days old: the download's age counts.
        self.set_link(linked_at=now_iso(), registry_fetched_at=av._iso(time.time() - 8 * 86400))
        sold = av.parse_registry(registry_zip([ROWS[0]]))  # only N100AC is still the company's
        with mock.patch.object(av, "registry", return_value=(sold, now_iso())) as reg:
            self.poll(T0)
        self.assertEqual(reg.call_count, 1)
        self.assertEqual(self.calls, ["a00001"])

    def test_old_registry_data_is_not_polled_when_it_cannot_be_rechecked(self):
        self.set_link(linked_at=now_iso(), registry_fetched_at=av._iso(time.time() - 15 * 86400))
        with mock.patch.object(av, "registry", side_effect=FetchError("FAA down")):
            self.poll(T0)
        self.assertEqual(self.calls, [])

    def test_links_from_an_older_name_rule_are_rechecked_first(self):
        self.set_link(rule=None)  # made before words were compared one by one
        with mock.patch.object(av, "registry", side_effect=FetchError("FAA down")):
            self.poll(T0)
        self.assertEqual(self.calls, [])  # not polled on the looser rule's word
        with mock.patch.object(av, "registry", return_value=(self.index, now_iso())):
            self.poll(T0 + 60)
        self.assertEqual(sorted(self.calls), ["a00001", "a00002", "a00042"])
        self.assertEqual(json.loads((config.home() / "aviation_links.json").read_text())["ACME"]["rule"],
                         av.MATCH_RULE)

    def test_partial_poll_keeps_the_watchlists_aircraft(self):
        st = config.home() / "aviation_state.json"
        links_p = config.home() / "aviation_links.json"
        links = json.loads(links_p.read_text())
        links["BETA"] = dict(links["ACME"], aircraft=[{"hex": "b00001", "reg": "N1BT"}, {"hex": "b00002", "reg": "N2BT"}])
        links_p.write_text(json.dumps(links))
        (config.home() / "aviation_optout.txt").write_text("N2BT\n")
        state = {"aircraft": {"b00001": {"phase": "airborne"}, "b00002": {"phase": "airborne"},
                              "c00001": {"phase": "ground"}}, "blocked": {}}
        st.write_text(json.dumps(state))
        (config.home() / "watchlist.json").write_text(json.dumps({"aircraft_of": ["BETA"]}))
        self.poll(T0)  # ACME by hand: BETA is still watched, c00001 is nobody's, N2BT opted out
        self.assertEqual(sorted(json.loads(st.read_text())["aircraft"]), ["a00001", "a00002", "a00042", "b00001"])
        (config.home() / "watchlist.json").write_text(json.dumps({"aircraft_of": []}))
        self.poll(T0 + 60)
        self.assertEqual(sorted(json.loads(st.read_text())["aircraft"]), ["a00001", "a00002", "a00042"])

    def test_no_companies(self):
        self.assertEqual(av.poll(self.bank, tickers=[], get=self.get, table=self.apts, log=lambda *_: None), 0)
        self.assertEqual(self.calls, [])


class ForgetTest(PollBase):
    """What was recorded before an opt-out or a LADD/PIA flag is withdrawn, alerts included."""

    def recorded(self, reg, hex_):
        """A landing as poll writes it, and the aircraft alert watch.check raises on it."""
        arrived = av._iso(T0)
        f = Fact(entity="ACME", attr="aircraft.flight", value="KTEB", as_of=arrived[:10], known_at=arrived,
                 tier="reported", source=dict(av.ADSB_SOURCE),
                 context={"hex": hex_, "reg": reg, "from": "KSJC", "to": "KTEB", "from_name": "San Jose Intl",
                          "to_name": "Teterboro", "departed": av._iso(T0 - 5 * 3600), "arrived": arrived},
                 retrieved_at=arrived)
        a = Fact(entity="ACME", attr="alert", value=f"ACME aircraft {reg} landed at KTEB (Teterboro) from KSJC",
                 as_of=arrived[:10], known_at=arrived, tier="derived", source={"name": "brain.watch", "rule": "aircraft"},
                 context={"rule": "aircraft", "severity": "notice", "keys": [f"aircraft:{hex_}:{arrived}"]},
                 inputs=(f.ref,), retrieved_at=arrived)
        self.bank.add_many([f, a])
        return f, a

    def test_optout_withdraws_registration_flights_and_alerts(self):
        f1, a1 = self.recorded("N100AC", "a00001")
        f2, a2 = self.recorded("N200AC", "a00002")
        self.assertEqual(av.add_optout("N100AC", self.bank), 3)
        for f in (f1, a1):
            self.assertTrue(self.bank.is_downgraded("ACME", f.id))
        for f in (f2, a2):
            self.assertFalse(self.bank.is_downgraded("ACME", f.id))
        self.assertEqual([f.id for f in self.bank.facts("ACME", "alert")], [a2.id])
        self.assertEqual([f.value for f in av.registered(self.bank.view(), "ACME")], ["N200AC", "N42AC"])
        self.assertEqual([f.id for f in av.flights(self.bank.view(), "ACME")], [f2.id])
        self.assertEqual(av.forget(self.bank), 0)  # done once

    def test_hand_edited_optout_is_applied_by_the_next_poll(self):
        f1, a1 = self.recorded("N100AC", "a00001")
        with (config.home() / "aviation_optout.txt").open("a") as fh:
            fh.write("a00001\n")
        self.poll(T0 + 60)
        self.assertTrue(self.bank.is_downgraded("ACME", f1.id))
        self.assertTrue(self.bank.is_downgraded("ACME", a1.id))

    def test_ladd_flag_withdraws_what_was_recorded_before(self):
        f2, a2 = self.recorded("N200AC", "a00002")
        self.answers = {"a00002": [{"hex": "a00002", "alt_baro": "ground", "lat": 37.3626, "lon": -121.929,
                                    "dbFlags": 8}]}
        self.poll(T0 + 60)
        self.assertTrue(self.bank.is_downgraded("ACME", f2.id))
        self.assertTrue(self.bank.is_downgraded("ACME", a2.id))
        self.assertTrue(self.bank.is_downgraded("ACME", next(
            f for f in self.bank.facts("ACME", "aircraft.registered", include_downgraded=True) if f.value == "N200AC").id))
        self.assertEqual(av.flights(self.bank.view(), "ACME"), [])

    def test_jets_link_lists_only_what_it_linked(self):
        with (config.home() / "aviation_optout.txt").open("a") as fh:
            fh.write("N200AC\n")  # opted out by hand after the link in setUp
        out = io.StringIO()
        with mock.patch.object(av, "registry", return_value=(self.index, now_iso())), \
                mock.patch.object(av, "Bank", return_value=self.bank), contextlib.redirect_stdout(out):
            av._cmd_link(argparse.Namespace(tickers=["ACME"]))
        text = out.getvalue()
        self.assertIn("2 aircraft registered", text)
        self.assertIn("N100AC", text)
        self.assertIn("N42AC", text)
        self.assertNotIn("N200AC", text)


class LiveTest(TempHome):
    def setUp(self):
        super().setUp()
        self._saved = (av._last_live, av._live_interval)
        av._last_live, av._live_interval = 0.0, av.ADSB_INTERVAL

    def tearDown(self):
        av._last_live, av._live_interval = self._saved
        super().tearDown()

    def answers(self, *codes):
        """urlopen stand-in: each call answers the next code (an HTTPError unless 200)."""
        import urllib.error
        seq = list(codes)

        def urlopen(req, timeout=None):
            code = seq.pop(0)
            if code != 200:
                raise urllib.error.HTTPError(req.full_url, code, "x", {"Retry-After": "2"}, None)
            return contextlib.nullcontext(io.BytesIO(b'{"ac": []}'))
        return urlopen

    def test_429_slows_down_and_retries_once(self):
        sleeps = []
        with mock.patch.object(av.urllib.request, "urlopen", self.answers(429, 200)), \
                mock.patch.object(av.time, "sleep", sleeps.append):
            self.assertEqual(av._live_json("https://api.adsb.lol/v2/hex/a00001"), {"ac": []})
        self.assertIn(2.0, sleeps)  # Retry-After honoured
        self.assertEqual(av._live_interval, 2 * av.ADSB_INTERVAL)
        with mock.patch.object(av.urllib.request, "urlopen", self.answers(429, 429)), \
                mock.patch.object(av.time, "sleep", sleeps.append):
            with self.assertRaises(av.FetchError):
                av._live_json("https://api.adsb.lol/v2/hex/a00001")
        self.assertEqual(av._live_interval, 4 * av.ADSB_INTERVAL)

    def test_other_errors_are_not_retried(self):
        with mock.patch.object(av.urllib.request, "urlopen", self.answers(404, 200)), \
                mock.patch.object(av.time, "sleep", lambda s: None):
            with self.assertRaises(av.FetchError):
                av._live_json("https://api.adsb.lol/v2/hex/a00001")
        self.assertEqual(av._live_interval, av.ADSB_INTERVAL)

    def test_pace_is_shared_with_other_processes(self):
        rate = config.home() / "rate.adsb.lol"
        before = time.time()
        rate.write_text(repr(before))  # another process asked adsb.lol just now
        sleeps = []
        with mock.patch.object(av.urllib.request, "urlopen", self.answers(200)), \
                mock.patch.object(av.time, "sleep", sleeps.append):
            av._live_json("https://api.adsb.lol/v2/hex/a00001")
        self.assertEqual(len(sleeps), 1)
        self.assertGreater(sleeps[0], 0.5)
        self.assertLessEqual(sleeps[0], av.ADSB_INTERVAL)
        self.assertGreaterEqual(float(rate.read_text()), before)  # our turn is written for the others


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
