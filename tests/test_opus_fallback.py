import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from scenario_model.classification import classify_features
from scenario_model.client import LlmClient, ProviderError
from scenario_model.config import (
    ACTIVE_MODELS, NUSVR_NO_OPUS_PROTOCOL_VERSION, NUSVR_PROTOCOL_VERSION,
    OPTIONAL_OPUS_MODEL, ROLE_FEATURE, ROOT, read_json,
)
from scenario_model.explanations import add_summary, read_summary, summary_request
from scenario_model.pipeline import evaluate_card, explain_score, replay_result
from scenario_model.scoring import ScenarioNuSVR
from test_pipeline import FakeClient


class FailingAssessorClient(FakeClient):
    def __init__(self, cache, model=OPTIONAL_OPUS_MODEL, stage="common"):
        super().__init__(cache)
        self.fail_model, self.fail_stage, self.failed_requests = model, stage, 0

    def _post(self, body):
        schema = body['response_format']['json_schema']
        props = schema['schema'].get('properties', {}).get('ratings', {}).get('items', {}).get('properties', {})
        stage = 'common' if 'main_character_status' in props else 'role_inputs' if ROLE_FEATURE in props else None
        if body['model'] == self.fail_model and stage == self.fail_stage:
            self.failed_requests += 1
            raise ProviderError('Synthetic assessor outage')
        raw = super()._post(body)
        if stage and body['model'] in ACTIVE_MODELS:
            payload = json.loads(raw['choices'][0]['message']['content'])
            row, i = payload['ratings'][0], ACTIVE_MODELS.index(body['model'])
            if stage == 'common':
                row['messages'] = ['Посыл ' + str(j) for j in range(i + 1)]
                row['message_competition_mean'] = i
            else:
                row[ROLE_FEATURE] = i
            raw['choices'][0]['message']['content'] = json.dumps(payload, ensure_ascii=False)
        return raw


class OpusFallbackTests(unittest.TestCase):
    def setUp(self):
        self.cache = tempfile.TemporaryDirectory()
        self.addCleanup(self.cache.cleanup)
        self.source = read_json(ROOT / 'examples/scenario.json')
        self.sleep = patch('scenario_model.client.time.sleep')
        self.sleep.start()
        self.addCleanup(self.sleep.stop)

    def test_opus_failure_calculates_score_class_and_contributions_from_three(self):
        client = FailingAssessorClient(self.cache.name)
        result = evaluate_card(self.source, client, use_nusvr=True)
        self.assertEqual(client.failed_requests, 3)
        self.assertEqual(result['protocol_version'], NUSVR_NO_OPUS_PROTOCOL_VERSION)
        self.assertEqual([a['model'] for a in result['assessors']], ACTIVE_MODELS[:3])
        self.assertEqual(len(result['observations']), 16)
        self.assertEqual(result['assessor_fallback']['reason'], 'provider_error')
        features = [a['features'] for a in result['assessors']]
        expected = ScenarioNuSVR().summarize(features, assessor_count=3)
        self.assertEqual(result['scoring']['assessor_scores'], expected['assessor_scores'])
        self.assertAlmostEqual(result['scoring']['score'], sum(expected['assessor_scores']) / 3)
        self.assertEqual(result['scoring']['feature_means']['message_competition_mean'], 1)
        self.assertEqual(result['scoring']['feature_means'][ROLE_FEATURE], 1)
        self.assertEqual(result['scoring']['classification'], classify_features(expected['feature_means'], joint=True))
        baseline, contributions = explain_score(result)
        self.assertAlmostEqual(baseline + sum(contributions.values()), result['scoring']['score'], places=9)
        self.assertEqual(replay_result(result), result['scoring'])

    def test_late_opus_failure_excludes_all_its_features_and_can_recover_from_cache(self):
        client = FailingAssessorClient(self.cache.name, stage='role_inputs')
        result = evaluate_card(self.source, client, use_nusvr=True)
        self.assertEqual(client.failed_requests, 3)
        self.assertTrue(any(r['model'] == OPTIONAL_OPUS_MODEL for r in client.calls))
        self.assertFalse(any(o['slot'].startswith(OPTIONAL_OPUS_MODEL) for o in result['observations']))
        self.assertEqual(result['scoring']['feature_means']['message_competition_mean'], 1)
        calls = len(client.calls)
        client.fail_model = None
        recovered = evaluate_card(self.source, client, use_nusvr=True)
        self.assertEqual(len(client.calls), calls + 1)
        self.assertEqual(recovered['protocol_version'], NUSVR_PROTOCOL_VERSION)
        self.assertNotIn('assessor_fallback', recovered)
        self.assertEqual(len(recovered['assessors']), 4)
        self.assertEqual(recovered['scoring']['feature_means']['message_competition_mean'], 1.5)
        self.assertEqual(replay_result(recovered), recovered['scoring'])
        self.assertEqual(replay_result(result), result['scoring'])

    def test_cached_fallback_and_cli_do_not_call_api(self):
        client = FailingAssessorClient(self.cache.name, stage='role_inputs')
        result = evaluate_card(self.source, client, use_nusvr=True)
        with patch('requests.post', side_effect=AssertionError('Unexpected API')):
            cached = evaluate_card(self.source, LlmClient(cache_dir=self.cache.name), use_nusvr=True)
            self.assertEqual(cached['scoring'], result['scoring'])
            self.assertEqual(cached['assessor_fallback']['reason'], 'missing_cached_observation')
            self.assertEqual(replay_result(cached), cached['scoring'])
        source_path = Path(self.cache.name) / 'input.json'
        output_path = Path(self.cache.name) / 'result.json'
        source_path.write_text(json.dumps(self.source, ensure_ascii=False))
        run = subprocess.run([sys.executable, '-m', 'scenario_model', 'evaluate', str(source_path), '--cache', self.cache.name, '--output', str(output_path)], cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        saved = json.loads(output_path.read_text())['results'][0]
        self.assertEqual(saved['protocol_version'], NUSVR_NO_OPUS_PROTOCOL_VERSION)
        self.assertEqual(replay_result(saved), saved['scoring'])

    def test_failure_of_another_assessor_or_legacy_ridge_still_stops_calculation(self):
        client = FailingAssessorClient(self.cache.name, model=ACTIVE_MODELS[1])
        with self.assertRaises(ProviderError):
            evaluate_card(self.source, client, use_nusvr=True)
        with tempfile.TemporaryDirectory() as cache:
            with self.assertRaises(ProviderError):
                evaluate_card(self.source, FailingAssessorClient(cache), use_function_demo=True)

    def test_fallback_metadata_version_observations_and_scores_are_checked(self):
        result = evaluate_card(self.source, FailingAssessorClient(self.cache.name), use_nusvr=True)
        variants = []
        for key, value in [('omitted_model', ACTIVE_MODELS[0]), ('assessor_count', 4), ('reason', 'unknown')]:
            bad = copy.deepcopy(result)
            bad['assessor_fallback'][key] = value
            variants.append(bad)
        bad = copy.deepcopy(result)
        del bad['assessor_fallback']
        variants.append(bad)
        bad = copy.deepcopy(result)
        bad['protocol_version'] = NUSVR_PROTOCOL_VERSION
        variants.append(bad)
        bad = copy.deepcopy(result)
        bad['observations'].pop()
        variants.append(bad)
        bad = copy.deepcopy(result)
        bad['scoring']['score'] += 1
        variants.append(bad)
        for bad in variants:
            with self.subTest(variant=variants.index(bad)), self.assertRaises(ValueError):
                replay_result(bad)

    def test_summary_and_streamlit_display_three_assessor_result(self):
        client = FailingAssessorClient(self.cache.name)
        result = evaluate_card(self.source, client, use_nusvr=True)
        self.assertIn('трёх оценок', summary_request(result, 'test')['messages'][0]['content'])
        add_summary(result, client)
        self.assertEqual(len(read_summary(result)), 9)
        self.sleep.stop()  # Streamlit's event loop needs its real polling sleep.
        with patch('scenario_model.pipeline.evaluate_card', return_value=copy.deepcopy(result)), patch('scenario_model.explanations.add_summary'), patch('requests.post', side_effect=AssertionError('Unexpected API')):
            at = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=20)
            at.text_area[0].set_value(json.dumps(self.source, ensure_ascii=False)).run()
            at.sidebar.text_input[0].set_value('fake-key').run()
            next(b for b in at.button if b.label == 'Оценить сценарии').click().run(timeout=20)
            self.assertEqual(len(at.exception), 0)
            self.assertEqual(len(at.error), 0)
            self.assertEqual(len(at.dataframe), 0)
            self.assertEqual(len(at.get('arrow_vega_lite_chart')), 1)
            self.assertTrue(any('трём оценщикам без Opus' in item.value for item in at.info))
            self.assertEqual(sum(x.value == 'Сводный вывод.' for x in at.text), 9)


if __name__ == '__main__':
    unittest.main()
