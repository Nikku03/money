import json
from unittest import mock

from brain.bank import Fact
from brain.sources import releases
from tests.util import TempHome

FOLDER = "https://www.sec.gov/Archives/edgar/data/123/000012345614000050/"
INDEX = """<table class="tableFile" summary="Document Format Files">
<tr><th scope="col">Seq</th><th scope="col">Description</th><th scope="col">Document</th><th scope="col">Type</th></tr>
<tr><td scope="row">1</td><td scope="row">FORM 8-K</td>
<td scope="row"><a href="/Archives/edgar/data/123/000012345614000050/form8k.htm">form8k.htm</a></td>
<td scope="row">8-K</td><td scope="row">21194</td></tr>
<tr class="evenRow"><td scope="row">2</td><td scope="row">CFO COMMENTARY</td>
<td scope="row"><a href="/Archives/edgar/data/123/000012345614000050/cfo.htm">cfo.htm</a></td>
<td scope="row">EX-99.2</td><td scope="row">9000</td></tr>
<tr><td scope="row">3</td><td scope="row">PRESS RELEASE</td>
<td scope="row"><a href="/Archives/edgar/data/123/000012345614000050/q2fy14pr.htm">q2fy14pr.htm</a></td>
<td scope="row">EX-99.1</td><td scope="row">849230</td></tr>
<tr><td scope="row">4</td><td scope="row"></td>
<td scope="row"><a href="/Archives/edgar/data/123/000012345614000050/logo.jpg">logo.jpg</a></td>
<td scope="row">GRAPHIC</td><td scope="row">3718</td></tr>
</table>"""
RELEASE = "<html><body><p>Acme Reports Second Quarter Results and Raises Full-Year Outlook</p></body></html>"


def fake_sec(url, max_bytes=None):
    pages = {FOLDER + "index.json": json.dumps({"directory": {"item": [
                 {"name": "0000123456-14-000050-index.html"}, {"name": "form8k.htm"}, {"name": "cfo.htm"},
                 {"name": "q2fy14pr.htm"}, {"name": "logo.jpg"}]}}),
             FOLDER + "0000123456-14-000050-index.html": INDEX,
             FOLDER + "q2fy14pr.htm": RELEASE}
    return pages[url].encode()


class Releases(TempHome):
    def filing(self):
        return Fact(entity="ACME", attr="filing", value="8-K", as_of="2014-07-30", known_at="2014-07-30T20:05:00Z",
                    tier="filed", source={"name": "SEC EDGAR", "url": FOLDER + "form8k.htm"},
                    context={"accn": "0000123456-14-000050", "items": "2.02,9.01"})

    def test_the_exhibit_from_its_name_or_else_from_the_index_types(self):
        self.assertEqual(releases.exhibit_name(["form8k.htm", "ex99-1.htm", "ex99-2.htm"]), "ex99-1.htm")
        self.assertIsNone(releases.exhibit_name(["form8k.htm", "q2fy14pr.htm"]))
        self.assertEqual(releases.typed_exhibit(INDEX), "q2fy14pr.htm")          # EX-99.1 before EX-99.2
        self.assertEqual(releases.typed_exhibit(INDEX.replace("EX-99.1", "EX-99")), "cfo.htm")   # else the first EX-99
        self.assertIsNone(releases.typed_exhibit(INDEX.replace("EX-99", "EX-10")))

    def test_an_empty_release_from_the_first_reader_is_read_again_and_superseded(self):
        f = self.filing()
        old = Fact(entity="ACME", attr=releases.ATTR, value="(no exhibit 99 in the filing)", as_of=f.as_of,
                   known_at=f.known_at, tier="filed", source={"name": "SEC EDGAR 8-K exhibit 99", "url": FOLDER},
                   context={"accn": "0000123456-14-000050", "exhibit": None, "events": [], "opening": "",
                            "opening_chars": releases.OPENING_CHARS})
        self.bank.add_many([f, old])
        view = self.bank.view()
        self.assertFalse(releases.settled(old))
        self.assertEqual(len(releases.pending(view, "ACME")[0]), 1)
        with mock.patch.object(releases, "_get", side_effect=fake_sec):
            out = releases.collect(self.bank, ["ACME"], log=None)
        self.assertEqual((out["kept"], out["failed"]), (1, []))
        live = self.bank.facts("ACME", releases.ATTR)
        self.assertEqual(len(live), 1)
        new = live[0]
        self.assertEqual((new.context["exhibit"], new.context["exhibit_by"]), ("q2fy14pr.htm", "index"))
        self.assertIn("guidance_raised", new.context["events"])
        self.assertTrue(self.bank.is_downgraded("ACME", old.id))
        self.assertEqual(releases.pending(self.bank.view(), "ACME"), ([], 1))
        with mock.patch.object(releases, "_get", side_effect=AssertionError("nothing left to read")):
            self.assertEqual(releases.collect(self.bank, ["ACME"], log=None)["kept"], 0)
