import unittest

from brain.terminal.commands import FUNCTIONS, CommandError, complete, parse


class ParseTest(unittest.TestCase):
    def check(self, line, fn, ticker, args=()):
        c = parse(line)
        self.assertEqual((c.fn, c.ticker, c.args), (fn, ticker, list(args)), line)
        self.assertEqual(c.raw, line)

    def test_ticker_function_args(self):
        self.check("AAPL GP 5Y", "GP", "AAPL", ["5Y"])
        self.check("aapl gp 5y spy qqq", "GP", "AAPL", ["5Y", "SPY", "QQQ"])
        self.check("AAPL CF 8-K", "CF", "AAPL", ["8-K"])

    def test_function_first(self):
        self.check("GP AAPL 5Y", "GP", "AAPL", ["5Y"])
        self.check("gp 5y aapl spy", "GP", "AAPL", ["5Y", "SPY"])  # a range is never the ticker
        self.check("CF 8-K AAPL", "CF", "AAPL", ["8-K"])  # nor is a form

    def test_bare_ticker_is_des(self):
        self.check("AAPL", "DES", "AAPL")
        self.check("  brk-b  ", "DES", "BRK-B")

    def test_noise_words(self):
        self.check("AAPL US EQUITY GP <GO>", "GP", "AAPL")
        self.check("AAPL US Equity GO", "DES", "AAPL")
        self.check("equities aapl fa", "FA", "AAPL")

    def test_market_symbols(self):
        self.check("^GSPC GP 5Y", "GP", "^GSPC", ["5Y"])
        self.check("EURUSD=X GP", "GP", "EURUSD=X")
        self.check("BTC-USD DES", "DES", "BTC-USD")
        self.check("000001.SS", "DES", "000001.SS")

    def test_tickers_that_are_function_codes(self):
        self.check("CF DES", "DES", "CF")
        self.check("CF CF 10-K", "CF", "CF", ["10-K"])
        self.check("GP GP 1Y", "GP", "GP", ["1Y"])

    def test_global_codes_as_tickers(self):
        # Movado, RPC, Okeanis Eco Tankers: a global code before a function that takes a ticker is the ticker.
        self.check("MOV DES", "DES", "MOV")
        self.check("RES GP 1Y", "GP", "RES", ["1Y"])
        self.check("ECO FA", "FA", "ECO")
        self.check("CLM DES", "DES", "CLM")
        self.check("MOV INGEST", "INGEST", "MOV")
        self.check("JOBS SHOW 1234abcd", "SHOW", "JOBS", ["1234ABCD"])
        self.check("GP MOV 5Y", "GP", "MOV", ["5Y"])
        self.check("ASK FA of AAPL?", "ASK", None, ["FA of AAPL?"])  # free text stays a question
        self.check("HELP GP", "HELP", None, ["GP"])

    def test_global_functions_take_no_arguments(self):
        for line, fn in (("MOV XYZ", "MOV"), ("WEI 5Y", "WEI"), ("MOV WEI", "MOV"), ("AAPL CURVE 1Y", "CURVE")):
            with self.assertRaises(CommandError, msg=line) as cm:
                parse(line)
            self.assertEqual((cm.exception.message, cm.exception.hint), (f"{fn} takes no arguments", fn))

    def test_go_is_noise_only_at_the_end(self):
        self.check("GO DES", "DES", "GO")  # Grocery Outlet
        self.check("GO", "DES", "GO")
        self.check("GO US EQUITY GP 5Y <GO>", "GP", "GO", ["5Y"])
        self.check("GO GO", "DES", "GO")
        self.check("DES GO", "DES", "GO")
        self.check("GP 5Y GO", "GP", "GO", ["5Y"])
        self.check("INGEST GO", "INGEST", "GO")
        self.check("AAPL GP 5Y GO", "GP", "AAPL", ["5Y"])
        self.check("INGEST MARKETS GO", "INGEST", None, ["MARKETS"])
        self.check("WEI GO", "WEI", None)

    def test_range_shaped_tokens_are_gp_arguments(self):
        self.check("GP 3Y AAPL", "GP", "AAPL", ["3Y"])  # data.run rejects the range; it is never the ticker
        self.check("AAPL GP 20Y", "GP", "AAPL", ["20Y"])
        with self.assertRaises(CommandError):
            parse("GP 3Y")

    def test_global_functions(self):
        self.check("WEI", "WEI", None)
        self.check("curve", "CURVE", None)
        self.check("HELP", "HELP", None)
        self.check("AAPL WEI", "WEI", "AAPL")  # a leading ticker is kept; global functions ignore it

    def test_ask_keeps_question_verbatim(self):
        self.check("ASK Why did AAPL's margin go up in FY2025?  ", "ASK", None,
                   ["Why did AAPL's margin go up in FY2025?"])
        self.check("AAPL US EQUITY ask is the P/E high? go", "ASK", "AAPL", ["is the P/E high? go"])
        self.check("ASK", "ASK", None)

    def test_show_and_ingest(self):
        self.check("AAPL SHOW 9de710d2", "SHOW", "AAPL", ["9DE710D2"])
        self.check("SHOW 9de710d2", "SHOW", None, ["9DE710D2"])
        self.check("SHOW AAPL 9de710d2", "SHOW", "AAPL", ["9DE710D2"])
        self.check("INGEST MARKETS", "INGEST", None, ["MARKETS"])
        self.check("INGEST nvda", "INGEST", "NVDA")
        self.check("NVDA INGEST", "INGEST", "NVDA")
        # First on the line, INGEST and SHOW are always the function, so the hint `INGEST CF` ingests CF.
        self.check("INGEST CF", "INGEST", "CF")
        self.check("INGEST MOV", "INGEST", "MOV")
        self.check("INGEST FA GP", "INGEST", "FA", ["GP"])
        self.check("INGEST AAPL CF", "INGEST", "AAPL", ["CF"])
        self.check("SHOW CF 1234abcd", "SHOW", "CF", ["1234ABCD"])
        self.check("CF SHOW 1234abcd", "SHOW", "CF", ["1234ABCD"])

    def test_errors_with_hints(self):
        with self.assertRaises(CommandError) as cm:
            parse("FA")
        self.assertEqual(cm.exception.message, "FA needs a ticker, e.g. AAPL FA")
        self.assertEqual(cm.exception.hint, "AAPL FA")
        with self.assertRaises(CommandError) as cm:
            parse("AAPL GPX")
        self.assertEqual(cm.exception.message, "unknown function 'GPX'")
        self.assertIn("GP", cm.exception.hint)
        with self.assertRaises(CommandError) as cm:
            parse("AAPL QWERTY")
        self.assertEqual(cm.exception.hint, "HELP")
        for bad in ("", "   ", "US EQUITY", "GP 5Y", "AAPL;rm GP", "INGEST", "AAPL SHOW"):
            with self.assertRaises(CommandError, msg=bad):
                parse(bad)

    def test_v2_functions(self):
        self.check("AAPL INS", "INS", "AAPL")
        self.check("ins aapl", "INS", "AAPL")
        self.check("NVDA US EQUITY ERN <GO>", "ERN", "NVDA")
        self.check("JET COST", "JET", "COST")
        self.check("SHIP", "SHIP", None)
        self.check("ship la-long-beach", "SHIP", None, ["LA-LONG-BEACH"])
        self.check("CAM TFL.00002.00865", "CAM", None, ["TFL.00002.00865"])
        self.check("CAM CALTRANS.D04.tvb04i880attruckscales", "CAM", None, ["CALTRANS.D04.TVB04I880ATTRUCKSCALES"])
        self.check("WL", "WL", None)
        self.check("WL ADD nvda aapl", "WL", None, ["ADD", "NVDA", "AAPL"])
        self.check("WL DEL GP", "WL", None, ["DEL", "GP"])  # a ticker that is a code, after WL's verb
        self.check("ALRT", "ALRT", None)
        self.check("ALRT ACK 1a2b3c4d", "ALRT", None, ["ACK", "1A2B3C4D"])
        self.check("ALRT ACK ALL", "ALRT", None, ["ACK", "ALL"])
        self.check("INGEST INSIDERS", "INGEST", None, ["INSIDERS"])
        self.check("INGEST WATCH", "INGEST", None, ["WATCH"])
        self.check("INGEST AAPL WATCH", "INGEST", "AAPL", ["WATCH"])
        # Seanergy Maritime trades as SHIP: [TICKER] FUNCTION still wins, and a bare SHIP is the function
        self.check("SHIP DES", "DES", "SHIP")
        self.check("SHIP GP 1Y", "GP", "SHIP", ["1Y"])
        self.check("INGEST SHIP", "INGEST", "SHIP")
        for line, hint in (("INS", "AAPL INS"), ("ERN", "AAPL ERN"), ("JET", "AAPL JET")):
            with self.assertRaises(CommandError) as cm:
                parse(line)
            self.assertEqual(cm.exception.hint, hint)

    def test_every_function_parses(self):
        for code, spec in FUNCTIONS.items():
            line = f"AAPL {code} 1234abcd" if code == "SHOW" else f"AAPL {code}"
            self.assertEqual(parse(line).fn, code)


class CompleteTest(unittest.TestCase):
    TICKERS = ["AAPL", "ABT", "AMD", "MACRO", "^GSPC", "SPY"]

    def texts(self, q):
        return [d["text"] for d in complete(q, self.TICKERS)]

    def test_tickers_and_functions(self):
        items = complete("a", self.TICKERS)
        self.assertIn({"text": "AAPL", "kind": "ticker", "desc": ""}, items)
        self.assertIn("ASK", self.texts("a"))
        self.assertNotIn("MACRO", self.texts("ma"))  # a bank entity, not a security
        self.assertIn("^GSPC", self.texts("^g"))
        self.assertIn("GC=F", self.texts("gold"))  # world-market names match too

    def test_after_a_ticker_suggests_functions(self):
        self.assertEqual(self.texts("AAPL G"), ["GP"])
        after = complete("AAPL ", self.TICKERS)
        self.assertTrue(after and all(d["kind"] == "function" for d in after))
        self.assertEqual(after[0]["text"], "DES")  # security functions before global ones
        self.assertIn("SPY", self.texts("AAPL GP 1Y S"))  # GP comparisons
        self.assertEqual(complete("GO ", self.TICKERS)[0]["text"], "DES")  # a leading GO is a ticker

    def test_v2_argument_completion(self):
        tickers = self.TICKERS + ["CAM.TFL.00002.00389", "AREA.TESTBOX", "AREA.LA-LONG-BEACH"]
        texts = lambda q: [d["text"] for d in complete(q, tickers)]
        self.assertNotIn("CAM.TFL.00002.00389", texts("c"))  # cameras and areas are not securities
        self.assertNotIn("AREA.TESTBOX", texts("a"))
        self.assertIn("CAM", texts("c"))
        self.assertEqual(texts("SHIP T"), ["TESTBOX"])
        self.assertIn("LA-LONG-BEACH", texts("SHIP "))  # sampled and built-in areas
        self.assertIn("HOUSTON", texts("SHIP H"))
        self.assertTrue(all(d["kind"] == "keyword" for d in complete("SHIP ", tickers)))
        self.assertEqual(texts("CAM "), ["TFL.00002.00389"])
        self.assertEqual(texts("SHIP TESTBOX "), [])  # one area at a time
        self.assertEqual(texts("WL "), ["ADD", "DEL"])
        self.assertEqual(texts("WL ADD A"), ["AAPL", "ABT", "AMD"])
        self.assertEqual(texts("ALRT "), ["ACK"])
        self.assertEqual(texts("ALRT ACK "), ["ALL"])
        self.assertEqual(texts("INGEST "), ["MARKETS", "MACRO", "UNIVERSE", "INSIDERS", "WATCH"])
        self.assertEqual(texts("INGEST I")[0], "INSIDERS")
        self.assertIn("INS", texts("AAPL I"))

    def test_limits(self):
        self.assertEqual(complete("", self.TICKERS), [])
        self.assertEqual(complete("ASK what about ", self.TICKERS), [])
        self.assertLessEqual(len(complete("AAPL ", self.TICKERS)), 12)
        self.assertEqual(self.texts("SPY")[0], "SPY")  # an exact match first
