# coding: utf-8
import copy, json, tempfile, unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from scenario_model.config import ROOT, read_json
from scenario_model.pipeline import evaluate_card
from test_pipeline import FakeClient


class AppTests(unittest.TestCase):
    def test_opening_and_entering_json_does_not_call_api(self):
        with patch(
            "requests.post", side_effect=AssertionError("Unexpected network")
        ) as post:
            at = AppTest.from_file(str(ROOT / "app.py")).run(timeout=20)
            self.assertEqual(len(at.exception), 0)
            at.text_area[0].set_value(
                json.dumps(
                    read_json(ROOT / "examples/scenario.json"), ensure_ascii=False
                )
            ).run()
            self.assertEqual(len(at.exception), 0)
            post.assert_not_called()

    def test_explicit_analysis_displays_claim_score_and_evidence(self):
        with tempfile.TemporaryDirectory() as cache:
            source = read_json(ROOT / "examples/scenario.json")
            result = evaluate_card(source, FakeClient(cache), use_function_demo=True)
        with patch(
            "scenario_model.pipeline.evaluate_card", return_value=copy.deepcopy(result)
        ) as evaluate, patch(
            "requests.post", side_effect=AssertionError("Unexpected network")
        ) as post:
            at = AppTest.from_file(str(ROOT / "app.py")).run(timeout=20)
            at.text_area[0].set_value(json.dumps(source, ensure_ascii=False)).run()
            at.sidebar.text_input[0].set_value("fake-key").run()
            next(b for b in at.button if b.label == "Оценить сценарии").click().run(
                timeout=20
            )
            self.assertEqual(len(at.exception), 0)
            evaluate.assert_called_once()
            self.assertTrue(evaluate.call_args.kwargs["use_function_demo"])
            post.assert_not_called()
            self.assertEqual(len(at.metric), 1)
            self.assertEqual(len(at.dataframe), 2)
            self.assertTrue(any("Выделенный посыл" in x.value for x in at.markdown))
            self.assertTrue(
                any("Показ ключевой функции сервиса/продукта" in x.value for x in at.markdown)
            )


if __name__ == "__main__":
    unittest.main()
