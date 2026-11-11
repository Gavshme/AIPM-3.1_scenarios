# coding: utf-8
import copy, json, tempfile, unittest, subprocess, sys, threading
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scenario_model.config import (
    ROOT, ACTIVE_MODELS, FEATURES, MODELS, FULL_CLAIM_PROTOCOL_VERSION, ORDINAL_CLASS_PROTOCOL_VERSION,
    canonical, read_json,
)
from scenario_model.cards import normalize_card, parse_cards, claim_input
from scenario_model.client import LlmClient, MissingObservation, ProviderError
from scenario_model.pipeline import evaluate_card, replay_result
from scenario_model.protocol import (
    protocols,
    request_body,
    validate_stage,
    select_claim,
    ProtocolError,
)
from scenario_model.scoring import ScenarioRidge

CLAIM = "Сервис доставляет лампу с проверкой до подтверждения оплаты."


def default(schema):
    if "enum" in schema:
        return schema["enum"][0]
    t = schema.get("type")
    t = next((x for x in t if x != "null"), None) if isinstance(t, list) else t
    if t == "object":
        return {k: default(schema["properties"][k]) for k in schema.get("required", [])}
    if t == "array":
        return []
    if t == "integer":
        return schema.get("minimum", 0)
    if t == "number":
        return schema.get("minimum", 0)
    if t == "boolean":
        return False
    return "Подтверждение в описании сцены."


def reply(body, claim=CLAIM):
    text = body["messages"][0]["content"]
    if body["response_format"]["json_schema"]["name"] == "feature_summary":
        summary_input = json.loads(text.split("\n\n", 1)[1])
        return {"features": [
            {"feature": row["feature"], "conclusion": "Сводный вывод.",
             "quote": summary_input["scenes"][0]["visual"]}
            for row in summary_input["features"]
        ]}
    marker = "\nSCENARIOS:\n" if "\nSCENARIOS:\n" in text else "\nСценарии:\n"
    card = json.loads(text.split(marker)[1])[0]
    schema = body["response_format"]["json_schema"]["schema"]
    row = default(schema["properties"]["ratings"]["items"])
    row["id"] = card["id"]
    if "target" in row:
        return {"ratings": [{"id": card["id"], "target": claim}]}
    if "score" in row and "scenes" in row:
        return {"ratings": [{"id": card["id"], "score": 2, "scenes": [2, 3],
                             "evidence": "Покупатель проверяет товар, затем подтверждает оплату."}]}
    if "message_text" in card:
        return {"ratings": [{
            "id": card["id"],
            "status": "formulated",
            "main_claim": claim,
            "source_quotes": [card["message_text"]],
            "removed_metadata": [],
            "reason": "Сохранено условие оплаты после проверки.",
        }]}
    if "main_claim" in row:
        row["main_claim"] = card.get("fixed_main_claim", claim)
    if "role_reveal_seconds" in row:
        row["benefits"] = []
        row["panel__distinct_benefit_count"] = 0
    elif "benefit_groups" in row:
        row.update(
            assessment_reason="Доставка и проверка до оплаты — две выгоды.",
            benefit_groups=[
                {
                    "group_id": 1,
                    "benefit": "Доставка",
                    "why_distinct": "Получение товара.",
                },
                {
                    "group_id": 2,
                    "benefit": "Проверка до оплаты",
                    "why_distinct": "Порядок проверки и оплаты.",
                },
            ],
            candidates=[
                {
                    "claim": "Доставка",
                    "source_path": "scenes[0].speech",
                    "quote": "Закажите лампу с доставкой",
                    "group_id": 1,
                    "decision_reason": "Названа доставка.",
                },
                {
                    "claim": "Проверка",
                    "source_path": "scenes[1].speech",
                    "quote": "Сначала проверьте покупку",
                    "group_id": 2,
                    "decision_reason": "Отдельная механика проверки.",
                },
            ],
        )
    elif "main_character_status" in row:
        row.update(
            main_character=1,
            main_character_status="confirmed",
            numeric_offer_on_screen=0,
            promo=0,
            panel__distinct_benefit_count=4,
            panel__offer_novelty_explanation_need=0,
            message_competition_mean=2,
            messages=["Покупка с доставкой и проверкой до оплаты"],
        )
    elif "panel__content_precision" in row:
        row["panel__content_precision"] = 4
    return {"ratings": [row]}


class FakeClient(LlmClient):
    def __init__(self, cache, *args, **kwargs):
        super().__init__(
            cache_dir=cache,
            api_key="TEST_SECRET_NOT_IN_CACHE",
            allow_live=True,
            *args,
            **kwargs
        )
        self.calls = []
        self.lock = threading.Lock()

    def _post(self, body):
        with self.lock:
            self.calls.append(copy.deepcopy(body))
        data = reply(body)
        return {
            "id": "fake",
            "model": body["model"],
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps(data, ensure_ascii=False)},
                }
            ],
        }


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = read_json(ROOT / "examples/scenario.json")
        self.client = FakeClient(self.tmp.name)

    def test_author_message_pipeline_and_replay(self):
        self.source["key_idea"] = "Не включать в запрос."
        result = evaluate_card(self.source, self.client, use_author_message=True)
        self.assertEqual(len(self.client.calls), 16)
        body = self.client.calls[0]["messages"][0]["content"]
        self.assertEqual(json.loads(body.split("\nSCENARIOS:\n")[1]), [{
            "id": self.source["id"], "message_text": self.source["message"]
        }])
        self.assertEqual(result["claim"]["fixed_main_claim"], CLAIM)
        self.assertEqual(result["claim"]["source"], "author_message")
        for body in self.client.calls[1:]:
            text = body["messages"][0]["content"]
            self.assertNotIn(self.source["message"], text)
            self.assertNotIn(self.source["key_idea"], text)
        self.assertEqual(replay_result(result), result["scoring"])
        del self.source["message"]
        with self.assertRaisesRegex(ValueError, "Сообщение"):
            evaluate_card(self.source, self.client, use_author_message=True)

    def test_demo_uses_full_claim_and_only_visual_scenes(self):
        result = evaluate_card(self.source, self.client, use_function_demo=True)
        self.assertEqual(len(self.client.calls), 17)
        self.assertEqual(result["protocol_version"], ORDINAL_CLASS_PROTOCOL_VERSION)
        self.assertEqual([row["model"] for row in result["assessors"]], ACTIVE_MODELS)
        self.assertEqual(set(body["model"] for body in self.client.calls), set(ACTIVE_MODELS))
        self.assertEqual(result["function_demo_target"], CLAIM)
        requests = {o["slot"]: o["request"] for o in result["observations"]}

        def card_from_request(slot):
            text = requests[slot]["messages"][0]["content"]
            return json.loads(text.split("\nSCENARIOS:\n")[1])[0]

        self.assertNotIn("demo_target/1", requests)
        for assessor in result["assessors"]:
            self.assertEqual(
                card_from_request(assessor["model"] + "/function_demo"),
                {
                    "id": self.source["id"],
                    "target": CLAIM,
                    "scenes": [
                        {"visual": scene["visual"]} for scene in self.source["scenes"]
                    ],
                },
            )
            self.assertEqual(
                assessor["features"]["panel__key_service_function_demonstration"], 2
            )
        self.assertEqual(replay_result(result), result["scoring"])
        result["function_demo_target"] = "Другой посыл"
        with self.assertRaisesRegex(ValueError, "Изменено предложение"):
            replay_result(result)

    def test_four_assessor_result_cannot_be_replayed_as_five_assessors(self):
        result = evaluate_card(self.source, self.client, use_function_demo=True)
        result["protocol_version"] = FULL_CLAIM_PROTOCOL_VERSION
        with self.assertRaisesRegex(ValueError, "21 этап"):
            replay_result(result)

    def test_ordinal_class_uses_its_own_prediction_and_preserves_ridge_score(self):
        result = evaluate_card(self.source, self.client, use_function_demo=True)
        scoring = result["scoring"]
        ridge = ScenarioRidge(read_json(ROOT / "artifacts/function_demo.json"))
        features = [a["features"] for a in result["assessors"]]
        ridge_scoring = ridge.summarize(features, assessor_count=4)
        self.assertEqual(scoring["score"], ridge_scoring["score"])
        self.assertEqual(scoring["assessor_scores"], ridge_scoring["assessor_scores"])
        classification = scoring["classification"]
        self.assertEqual(scoring["class"], classification["class"])
        self.assertAlmostEqual(sum(classification["probabilities"].values()), 1)
        classification["probabilities"]["Низкий"] += 0.01
        with self.assertRaisesRegex(ValueError, "Данные классификатора"):
            replay_result(result)

    def test_end_to_end_blind_claim_same_five_and_cached_replay(self):
        self.source.update(
            fixed_main_claim="AUTHOR_SECRET_OLD_CLAIM",
            message="AUTHOR_SECRET_MESSAGE",
            key_idea="AUTHOR_SECRET_IDEA",
        )
        result = evaluate_card(self.source, self.client)
        self.assertEqual(len(self.client.calls), 18)
        self.assertEqual(result["claim"]["fixed_main_claim"], CLAIM)
        self.assertEqual([a["model"] for a in result["assessors"]], MODELS)
        for body in self.client.calls:
            text = body["messages"][0]["content"]
            self.assertNotIn("AUTHOR_SECRET", text)
            if (
                "role_reveal_seconds"
                in body["response_format"]["json_schema"]["schema"]["properties"][
                    "ratings"
                ]["items"]["properties"]
            ):
                data = json.loads(text.split("\nСценарии:\n")[1])[0]
                self.assertEqual(set(data), {"id", "duration_sec", "scenes"})
                self.assertNotIn("voice_and_music", data)
            else:
                self.assertEqual(
                    json.loads(text.split("\nSCENARIOS:\n")[1])[0]["fixed_main_claim"],
                    CLAIM,
                )
        self.assertEqual(len({o["fingerprint"] for o in result["observations"][:3]}), 3)
        for a in result["assessors"]:
            self.assertEqual(a["features"]["panel__distinct_benefit_count"], 2)
            self.assertEqual(a["features"]["panel__content_precision"], 4)
            self.assertEqual(a["features"]["message_competition_mean"], 0)
        self.assertEqual(replay_result(result)["score"], result["scoring"]["score"])
        again = evaluate_card(self.source, self.client)
        self.assertEqual(again["scoring"], result["scoring"])
        self.assertEqual(len(self.client.calls), 18)
        for file in Path(self.tmp.name).rglob("*.json"):
            self.assertNotIn("TEST_SECRET_NOT_IN_CACHE", file.read_text())

    def test_missing_cache_never_calls_provider(self):
        client = LlmClient(cache_dir=self.tmp.name, api_key="x", allow_live=False)
        with patch("requests.post", side_effect=AssertionError("network")) as post:
            with self.assertRaises(MissingObservation):
                evaluate_card(self.source, client)
            post.assert_not_called()

    def test_cache_input_change_requires_new_observations(self):
        evaluate_card(self.source, self.client)
        changed = copy.deepcopy(self.source)
        changed["scenes"][0]["speech"] += " Дополнительная реплика."
        with self.assertRaises(MissingObservation):
            evaluate_card(changed, LlmClient(cache_dir=self.tmp.name, allow_live=False))

    def test_author_fields_do_not_change_model_requests(self):
        a = evaluate_card(self.source, self.client)
        changed = copy.deepcopy(self.source)
        changed["message"] = "Новый авторский посыл."
        b = evaluate_card(changed, self.client)
        self.assertEqual(a["scoring"], b["scoring"])
        self.assertEqual(len(self.client.calls), 18)

    def test_saved_result_tampering_is_rejected(self):
        a = evaluate_card(self.source, self.client)
        a["scoring"]["score"] += 1
        with self.assertRaises(ValueError):
            replay_result(a)
        a = evaluate_card(self.source, self.client)
        a["observations"][0]["response"]["ratings"][0]["main_claim"] = "Другой посыл"
        with self.assertRaises(ValueError):
            replay_result(a)

    def test_claim_explanations_cannot_be_changed_in_saved_result(self):
        result = evaluate_card(self.source, self.client)
        result["claim_evidence"][0] = "Подменённое обоснование."
        with self.assertRaises(ValueError):
            replay_result(result)

    def test_literal_benefit_quotes_are_required(self):
        c, _ = normalize_card(self.source)
        spec = protocols()["assessors"][0]["stages"]["benefits"]
        body = request_body(spec, c, CLAIM)
        payload = reply(body)
        payload["ratings"][0]["candidates"][0]["quote"] = "Доставка за пять минут"
        with self.assertRaises(ProtocolError):
            validate_stage("benefits", spec, c, payload, CLAIM)

    def test_changed_assessor_claim_is_rejected(self):
        c, _ = normalize_card(self.source)
        spec = protocols()["assessors"][0]["stages"]["common"]
        payload = reply(request_body(spec, c, CLAIM))
        payload["ratings"][0]["main_claim"] = "Другой тезис"
        with self.assertRaises(ProtocolError):
            validate_stage("common", spec, c, payload, CLAIM)

    def test_claim_selection_is_modal_and_tie_is_first(self):
        self.assertEqual(select_claim(["a", "b", "b"])["fixed_main_claim"], "b")
        self.assertEqual(select_claim(["a", "b", "c"])["fixed_main_claim"], "a")
        self.assertTrue(select_claim(["a", "b", "c"])["all_different"])

    def test_competition_pooled_before_nonlinear_predictions(self):
        rows = [dict(zip(FEATURES, [1, 0, 2, 0, 0, 4, i])) for i in range(5)]
        model = ScenarioRidge()
        score = model.summarize(rows)
        self.assertEqual(
            [r["message_competition_mean"] for r in score["model_inputs"]], [2] * 5
        )
        self.assertEqual(score["feature_ranges"]["message_competition_mean"], [0, 4])
        self.assertAlmostEqual(
            score["score"],
            float(np.mean(model.predict(score["model_inputs"])) * 100),
            places=12,
        )

    def test_partial_failure_produces_no_score(self):
        class Failing(FakeClient):
            def _post(inner, body):
                if body["model"] == MODELS[-1]:
                    raise ProviderError("fake failure")
                return super(Failing, inner)._post(body)

        client = Failing(self.tmp.name)
        with patch("scenario_model.client.time.sleep"):
            with self.assertRaises(ProviderError):
                evaluate_card(self.source, client)
        self.assertGreater(len(list(Path(self.tmp.name).rglob("*.json"))), 3)

    def test_cli_uses_cache_and_replays_without_key(self):
        evaluate_card(self.source, self.client, use_function_demo=True)
        input = Path(self.tmp.name) / "input.json"
        output = Path(self.tmp.name) / "output.json"
        input.write_text(json.dumps(self.source, ensure_ascii=False))
        p = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "scenario_model",
                "evaluate",
                str(input),
                "--cache",
                self.tmp.name,
                "--output",
                str(output),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        p = subprocess.run(
            [sys.executable, "-B", "-m", "scenario_model", "replay", str(output)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(p.returncode, 0, p.stderr)

    def test_duplicate_ids_and_bad_intervals_rejected(self):
        with self.assertRaises(ValueError):
            parse_cards(json.dumps([self.source, self.source]))
        s = copy.deepcopy(self.source)
        s["scenes"][1]["time"] = "4–20 с"
        with self.assertRaises(ValueError):
            normalize_card(s)


if __name__ == "__main__":
    unittest.main()
