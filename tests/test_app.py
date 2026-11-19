# coding: utf-8
import copy, json, tempfile, unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from scenario_model.config import LABELS, ROOT, read_json
from scenario_model.explanations import FEATURE_DESCRIPTIONS, add_summary, contribution_display
from scenario_model.pipeline import evaluate_card, explain_score
from test_pipeline import FakeClient


class AppTests(unittest.TestCase):
    def test_collapsed_results_show_name_and_score_in_descending_unrounded_order(self):
        with tempfile.TemporaryDirectory() as cache:
            source = read_json(ROOT / "examples/scenario.json")
            result = evaluate_card(source, FakeClient(cache), use_nusvr=True)
        items, results = [], []
        for identifier, name, score in [("low", "Низкий скор", 18.2), ("middle", "Средний скор", 27.41), ("high", "Высокий скор", 27.44)]:
            item = {**copy.deepcopy(source), "id": identifier, "name": name}
            row = copy.deepcopy(result)
            row["input_card"] = item
            row["card"]["id"] = identifier
            row["scoring"]["score"] = score
            items.append(item)
            results.append(row)
        with patch("scenario_model.pipeline.evaluate_card", side_effect=results), patch("scenario_model.explanations.add_summary"), patch("requests.post", side_effect=AssertionError("Unexpected network")):
            at = AppTest.from_file(str(ROOT / "app.py")).run(timeout=20)
            at.text_area[0].set_value(json.dumps(items, ensure_ascii=False)).run()
            at.sidebar.text_input[0].set_value("fake-key").run()
            next(b for b in at.button if b.label == "Оценить сценарии").click().run(timeout=20)
            self.assertEqual(len(at.exception), 0)
            rows = [x for x in at.expander if " · MD " in x.label]
            self.assertEqual([x.label for x in rows], ["Высокий скор · MD 27,4", "Средний скор · MD 27,4", "Низкий скор · MD 18,2"])
            self.assertTrue(all(not x.proto.expanded for x in rows))

    def test_batch_limit_allows_20_and_blocks_21_before_api_calls(self):
        source = read_json(ROOT / "examples/scenario.json")
        with patch("requests.post", side_effect=AssertionError("Unexpected network")) as post:
            at = AppTest.from_file(str(ROOT / "app.py")).run(timeout=20)
            self.assertTrue(any("не больше 20 сценариев" in c.value for c in at.caption))
            for count in (20, 21):
                cards = [{**copy.deepcopy(source), "id": f"batch-{i}"} for i in range(count)]
                at.text_area[0].set_value(json.dumps(cards, ensure_ascii=False)).run()
                self.assertEqual(len(at.exception), 0)
                buttons = [b for b in at.button if b.label in {"Оценить сценарии", "Рассчитать из сохранённых ответов"}]
                self.assertEqual([b.disabled for b in buttons], [count > 20, count > 20])
                self.assertEqual(len(at.error), int(count > 20))
                if count > 20:
                    self.assertIn("не больше 20 сценариев", at.error[0].value)
            post.assert_not_called()

    def test_opening_and_entering_json_does_not_call_api(self):
        with patch(
            "requests.post", side_effect=AssertionError("Unexpected network")
        ) as post:
            at = AppTest.from_file(str(ROOT / "app.py")).run(timeout=20)
            self.assertEqual(len(at.exception), 0)
            requirements = next(x.value for x in at.markdown if "Требования к файлу" in x.value)
            for section in ["Сценарий", "Голос и Музыка", "Звуковой дизайн", "Креативные приёмы", "Сообщение"]:
                self.assertIn(f"**{section}**", requirements)
            at.text_area[0].set_value(
                json.dumps(
                    read_json(ROOT / "examples/scenario.json"), ensure_ascii=False
                )
            ).run()
            self.assertEqual(len(at.exception), 0)
            post.assert_not_called()

    def test_explicit_analysis_displays_claim_score_and_evidence(self):
        cache = tempfile.TemporaryDirectory()
        self.addCleanup(cache.cleanup)
        source = read_json(ROOT / "examples/scenario.json")
        fake = FakeClient(cache.name)
        result = evaluate_card(source, fake, use_function_demo=True, use_nusvr=True)
        baseline, contributions = explain_score(result)
        self.assertAlmostEqual(baseline + sum(contributions.values()), result["scoring"]["score"])
        self.assertEqual(len(result["scoring"]["assessor_scores"]), 4)
        self.assertAlmostEqual(result["scoring"]["score"], sum(result["scoring"]["assessor_scores"]) / 4)
        with patch(
            "scenario_model.pipeline.evaluate_card", return_value=copy.deepcopy(result)
        ) as evaluate, patch("scenario_model.explanations.add_summary",
            side_effect=lambda result, client, model: add_summary(result, fake, model)
        ), patch(
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
            self.assertTrue(evaluate.call_args.kwargs["use_nusvr"])
            post.assert_not_called()
            self.assertEqual(len(at.dataframe), 0)
            self.assertEqual(len(at.get("arrow_vega_lite_chart")), 0)
            self.assertIn("Основной посыл", at.get("html")[0].proto.body)
            self.assertIn("Класс по признакам сценария", at.get("html")[0].proto.body)
            self.assertIn(f"{result['scoring']['classification']['class']} класс", at.get("html")[0].proto.body)
            self.assertIn(f"{result['scoring']['score']:.1f}", at.get("html")[0].proto.body)
            self.assertEqual(sum(x.value == "Сводный вывод." for x in at.text), 9)
            self.assertEqual(sum("В сценарии" in x.value for x in at.caption), 9)
            descriptions = [x.value for x in at.caption if x.value.startswith("Что оцениваем:")]
            self.assertEqual(descriptions, ["Что оцениваем: " + FEATURE_DESCRIPTIONS[f] for f in result["scoring"]["feature_means"]])
            impacts = [x.value for x in at.markdown if any(f"**{label}**" in x.value for label in ["Положительный вклад", "Отрицательный вклад", "Нейтральный вклад"])]
            expected_impacts = []
            for feature in result["scoring"]["feature_means"]:
                label, color, number = contribution_display(contributions[feature])
                expected_impacts.append(f":{color}[**{label}**]")
            self.assertEqual(impacts, expected_impacts)
            for direction in ["Положительный вклад", "Отрицательный вклад"]:
                features = [f for f in result["scoring"]["feature_means"] if contribution_display(contributions[f])[0] == direction]
                if features:
                    self.assertIn("\n".join("- " + LABELS[f] for f in features), [x.value for x in at.markdown])
            self.assertTrue(any("Повышают скор" in x.value for x in at.markdown))
            self.assertTrue(any("Понижают скор" in x.value for x in at.markdown))
            self.assertFalse(any("п.п." in x.value for x in [*at.markdown, *at.caption]))
            blocks = [node for node in at.get("flex_container") if node.proto.flex_container.border]
            self.assertEqual(len(blocks), 9)
            self.assertEqual(len(fake.calls), 22)
            self.assertTrue(
                any("Показ ключевой функции сервиса/продукта" in x.value for x in at.markdown)
            )


if __name__ == "__main__":
    unittest.main()
