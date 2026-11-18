import copy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scenario_model.classification import CLASSES, classify_features
from scenario_model.config import NUSVR_FEATURES, NUSVR_PROTOCOL_VERSION, ROLE_FEATURE, ROOT, read_json
from scenario_model.pipeline import evaluate_card, explain_score, replay_result
from scenario_model.scoring import ScenarioNuSVR
from test_pipeline import FakeClient


class NuSVRReleaseTests(unittest.TestCase):
    def test_synthetic_export_parity_and_mode_decision(self):
        model = ScenarioNuSVR()
        mode_differs_from_median = False
        for fixture in read_json(ROOT / "tests/fixtures/nusvr_ordinal.json"):
            scoring = model.summarize(fixture["rows"], assessor_count=4)
            np.testing.assert_allclose(scoring["assessor_scores"], fixture["assessor_scores"], atol=1e-10, rtol=0)
            self.assertAlmostEqual(scoring["score"], fixture["mean_score"], places=10)
            classification = classify_features(scoring["feature_means"], joint=True)
            self.assertEqual(classification["class"], CLASSES[fixture["class_index"]])
            np.testing.assert_allclose(list(classification["probabilities"].values()), fixture["class_probabilities"], atol=1e-12, rtol=0)
            p = np.asarray(fixture["class_probabilities"])
            mode_differs_from_median |= int(p.argmax()) != int((np.cumsum(p) < .5).sum())
        self.assertTrue(mode_differs_from_median)

    def test_ninth_score_feature_does_not_change_the_eight_feature_class(self):
        rows = copy.deepcopy(read_json(ROOT / "tests/fixtures/nusvr_ordinal.json")[0]["rows"])
        model = ScenarioNuSVR()
        for row in rows:
            row[ROLE_FEATURE] = 0
        early = model.summarize(rows, assessor_count=4)
        for row in rows:
            row[ROLE_FEATURE] = 4
        late = model.summarize(rows, assessor_count=4)
        self.assertNotAlmostEqual(early["score"], late["score"], places=6)
        self.assertEqual(classify_features(early["feature_means"], joint=True), classify_features(late["feature_means"], joint=True))

    def test_current_pipeline_replay_explanations_and_cache_reuse(self):
        with tempfile.TemporaryDirectory() as cache:
            client = FakeClient(cache)
            source = read_json(ROOT / "examples/scenario.json")
            legacy = evaluate_card(source, client, use_function_demo=True)
            self.assertEqual(len(client.calls), 17)
            with patch("requests.post", side_effect=AssertionError("Unexpected live API")):
                result = evaluate_card(source, client, use_nusvr=True)
                self.assertEqual(len(client.calls), 21)  # Reuse the existing 17 stages, add four new readings.
                self.assertEqual(result["protocol_version"], NUSVR_PROTOCOL_VERSION)
                self.assertEqual(result["model_id"], "scenario-nusvr9-four-20261118")
                self.assertEqual(list(result["scoring"]["feature_means"]), NUSVR_FEATURES)
                self.assertEqual(replay_result(result), result["scoring"])
                self.assertEqual(replay_result(legacy), legacy["scoring"])
                again = evaluate_card(source, client, use_nusvr=True)
                self.assertEqual(again["scoring"], result["scoring"])
                self.assertEqual(len(client.calls), 21)
                baseline, contributions = explain_score(result)
                self.assertAlmostEqual(baseline + sum(contributions.values()), result["scoring"]["score"], places=9)
            missing = copy.deepcopy(result)
            missing["observations"] = [o for o in missing["observations"] if not o["slot"].endswith("role_inputs")]
            with self.assertRaisesRegex(ValueError, "21 этап"):
                replay_result(missing)
            altered = copy.deepcopy(result)
            altered["scoring"]["classification"]["probabilities"]["Средний"] += .01
            with self.assertRaisesRegex(ValueError, "Данные классификатора"):
                replay_result(altered)

    def test_missing_role_never_uses_an_imputed_zero(self):
        rows = copy.deepcopy(read_json(ROOT / "tests/fixtures/nusvr_ordinal.json")[0]["rows"])
        del rows[0][ROLE_FEATURE]
        with self.assertRaises(KeyError):
            ScenarioNuSVR().summarize(rows, assessor_count=4)


if __name__ == "__main__":
    unittest.main()
