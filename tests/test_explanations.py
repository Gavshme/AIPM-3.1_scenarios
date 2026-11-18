import copy
import json
import tempfile
import unittest

from scenario_model.config import NUSVR_FEATURES, ROOT, read_json
from scenario_model.explanations import (
    FEATURE_DESCRIPTIONS, LEGACY_SLOT, SLOT, add_summary, clean_final_comment,
    contribution_display, final_comment, read_summary, summary_request, validate_summary,
)
from scenario_model.pipeline import evaluate_card, replay_result
from test_pipeline import FakeClient


class ExplanationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = tempfile.TemporaryDirectory()
        cls.client = FakeClient(cls.cache.name)
        cls.result = evaluate_card(read_json(ROOT / 'examples/scenario.json'), cls.client, use_nusvr=True)

    @classmethod
    def tearDownClass(cls):
        cls.cache.cleanup()

    def test_definition_coverage_and_contribution_signs(self):
        self.assertEqual(set(FEATURE_DESCRIPTIONS), set(NUSVR_FEATURES))
        for value, expected in [(1.234, ('Положительный вклад', 'green', '+1.23')),
                                (-2.345, ('Отрицательный вклад', 'red', '−2.35')),
                                (0, ('Нейтральный вклад', 'gray', '0')),
                                (-0.001, ('Нейтральный вклад', 'gray', '0'))]:
            self.assertEqual(contribution_display(value), expected)

    def test_new_summary_uses_mean_and_requests_one_final_comment(self):
        body = summary_request(self.result, 'test')
        prompt, serialized = body['messages'][0]['content'].split('\n\n', 1)
        self.assertIn('один итоговый комментарий', prompt)
        self.assertIn('Не упоминай разногласия', prompt)
        self.assertNotIn('кратко назови его причину', prompt)
        for row in json.loads(serialized)['features']:
            self.assertEqual(row['mean_score'], self.result['scoring']['feature_means'][row['feature']])

    def test_disagreement_comments_are_rejected_for_new_summaries(self):
        payload = {'features': [dict(feature=f, conclusion='В сценарии показан выбор товара.', quote='') for f in NUSVR_FEATURES]}
        for sentence in ['Оценки различаются.', 'Есть разногласия между моделями.', 'Оценщики не согласны.', 'Расхождение оценок связано с описанием.']:
            bad = copy.deepcopy(payload)
            bad['features'][0]['conclusion'] = sentence
            with self.subTest(sentence=sentence), self.assertRaises(ValueError):
                validate_summary(self.result, bad)
        self.assertEqual(len(validate_summary(self.result, payload)[0]), 9)

    def test_cleaning_preserves_facts_and_falls_back_when_only_meta_is_present(self):
        self.assertEqual(clean_final_comment('Оценки различаются. Пользователь выбирает товар в приложении.'), 'Пользователь выбирает товар в приложении.')
        feature = NUSVR_FEATURES[0]
        text = final_comment(self.result, feature, 'Оценщики не согласны.')
        self.assertTrue(text)
        self.assertNotIn('Оценщики', text)
        self.assertEqual(clean_final_comment('Механика оплаты описана неоднозначно.'), 'Механика оплаты описана неоднозначно.')

    def test_legacy_summaries_still_verify_without_new_api_calls(self):
        result = copy.deepcopy(self.result)
        body = summary_request(result, 'test-summary-model', legacy=True)
        # Existing saved observations used this slot and lacked aggregate inputs.
        self.assertNotIn('mean_score', json.loads(body['messages'][0]['content'].split('\n\n', 1)[1])['features'][0])
        _, _, observation = self.client.observe(body, LEGACY_SLOT, lambda payload: validate_summary(result, payload, require_final=False))
        result['summary_observation'] = observation
        calls = len(self.client.calls)
        before = copy.deepcopy(result)
        self.assertEqual(len(read_summary(result)), 9)
        self.assertEqual(len(self.client.calls), calls)
        self.assertEqual(result, before)
        self.assertEqual(replay_result(result), result['scoring'])

    def test_new_summary_keeps_score_and_class_and_does_not_reuse_old_slot(self):
        result = copy.deepcopy(self.result)
        original_scoring = copy.deepcopy(result['scoring'])
        add_summary(result, self.client, model='test-summary-model')
        self.assertEqual(result['summary_observation']['slot'], SLOT)
        self.assertNotEqual(SLOT, LEGACY_SLOT)
        self.assertEqual(len(read_summary(result)), 9)
        self.assertEqual(result['scoring'], original_scoring)
        self.assertEqual(replay_result(result), original_scoring)


if __name__ == '__main__':
    unittest.main()
