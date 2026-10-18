import json

from brain import chain
from brain.chain import distill
from tests.util import TempHome

TEXTBOOK = [{"cause": "oil_price_up", "effect": "airline_fuel_costs_up", "sign": 1},
            {"cause": "airline_fuel_costs_up", "effect": "airline_profits_down", "sign": 1},
            {"cause": "airline_profits_down", "effect": "airline_stocks_down", "sign": 1},
            {"cause": "oil_price_up", "effect": "oil_producer_stocks_up", "sign": 1}]


class FakeClient:
    model = "fake-model"

    def __init__(self, answers):
        self.answers, self.asked = answers, []

    def complete(self, messages, **kw):
        cause = messages[1]["content"].split("Market state: ")[1].split(".\n")[0]
        self.asked.append(cause)
        return {"choices": [{"message": {"content": self.answers.get(cause, '{"effects": []}')}}]}


class DistillTest(TempHome):
    def test_parse_keeps_known_states_and_clean_signs(self):
        names = distill.states(TEXTBOOK)
        got, why = distill.parse('<think>hmm</think>```json\n{"effects": [{"state": "Airline Fuel Costs Up", "sign": "+1"},'
                                 '{"state": "unicorns_up", "sign": 1}, {"state": "oil_price_up", "sign": 1},'
                                 '{"state": "airline_fuel_costs_up", "sign": -1}]}\n```', "oil_price_up", names)
        self.assertEqual((got, why), ([{"cause": "oil_price_up", "effect": "airline_fuel_costs_up", "sign": 1}], ""))
        self.assertEqual(distill.parse("no idea", "oil_price_up", names), ([], "the answer holds no JSON object"))

    def test_ask_compare_and_report(self):
        answers = {"oil_price_up": json.dumps({"effects": [{"state": "airline_fuel_costs_up", "sign": 1},
                                                           {"state": "oil_producer_stocks_up", "sign": -1}]}),
                   "airline_fuel_costs_up": '{"effects": [{"state": "airline_stocks_down", "sign": 1}]}',
                   "airline_profits_down": "not json"}
        client = FakeClient(answers)
        table = distill.ask(client, TEXTBOOK, log=None)
        self.assertEqual(client.asked, distill.states(TEXTBOOK))   # every state asked once, nothing else given
        self.assertTrue(distill.path_for("fake-model").exists())
        self.assertEqual(table["failed"], [{"cause": "airline_profits_down", "why": "the answer holds no JSON object"}])
        results = {"L1-oil-producers-same": {"metrics": {"test": {"per_move": 0.03, "p": 0.0002}}, "verdict": "REPORT"},
                   "L1-oil-producers-next": {"verdict": "FAIL"}}
        net = chain.train(steps=40, log=None)
        cmp = distill.compare(table["links"], TEXTBOOK, results, net)
        self.assertEqual((cmp["recall"]["found"], cmp["recall"]["same_sign"]), (2, 1))
        self.assertEqual(cmp["recall"]["wrong_sign"], ["oil_price_up -> oil_producer_stocks_up"])
        self.assertEqual(cmp["recall"]["added"], ["airline_fuel_costs_up -> airline_stocks_down (+1)"])
        self.assertEqual(cmp["reasoning"]["followed"], cmp["reasoning"]["chains"])
        rows = {r["pair"]: r for r in cmp["reality"]}
        prod = rows["oil_price_up -> oil_producer_stocks_up"]
        self.assertEqual(prod["model_sign"], -1)                    # the model says producers fall with oil
        self.assertTrue(prod["same_month"].startswith("the data says the opposite"))
        self.assertEqual(prod["next_month"], "tested only the other way: FAIL")
        air = rows["oil_price_up -> airline_stocks_down"]
        self.assertTrue(air["same_month"].startswith("untested"))   # no result given for it here
        text = distill.report(table, cmp)
        self.assertIn("1. Recall: 2 of the 4 textbook links, 1 with the right sign", text)
