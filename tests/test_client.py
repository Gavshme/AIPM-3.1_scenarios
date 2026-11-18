import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scenario_model.client import LlmClient, ProviderError, ResponseFormatError, decode_response


PAYLOAD = {"ratings": [{"id": "S000", "main_character": 1,
                         "evidence": 'Текст содержит кавычки "да", скобки [1] и запятую ,}.'}]}


def response(content, finish="stop"):
    return {"id": "example", "model": "anthropic/claude-opus-4.6",
            "choices": [{"finish_reason": finish, "message": {"content": content}}]}


class ClientResponseTests(unittest.TestCase):
    def test_plain_fenced_and_prefixed_json_preserve_values(self):
        text = json.dumps(PAYLOAD, ensure_ascii=False)
        for wrapped in [text, "```json\n" + text + "\n```",
                        "Результат:\n```JSON\n" + text + "\n```\nГотово.",
                        "\ufeff" + text]:
            with self.subTest(wrapped=wrapped[:20]):
                self.assertEqual(decode_response(response(wrapped))[0], PAYLOAD)

    def test_text_blocks_are_concatenated(self):
        text = json.dumps(PAYLOAD)
        blocks = [{"type": "text", "text": text[:25]}, {"type": "text", "text": text[25:]}]
        self.assertEqual(decode_response(response(blocks))[0], PAYLOAD)

    def test_trailing_commas_and_literal_newlines_preserve_evidence(self):
        text = '{"ratings":[{"id":"S000","main_character":1,"evidence":"Строка 1\nСтрока 2 ,}",},],}'
        payload, notes = decode_response(response(text))
        self.assertEqual(payload["ratings"][0]["main_character"], 1)
        self.assertEqual(payload["ratings"][0]["evidence"], "Строка 1\nСтрока 2 ,}")
        self.assertTrue(notes)

    def test_incomplete_ambiguous_or_invalid_answers_are_not_repaired(self):
        for text in ['', None, 'Сбой сервиса', '{"ratings":[' + json.dumps(PAYLOAD['ratings'][0]),
                     json.dumps(PAYLOAD) + '\n' + json.dumps(PAYLOAD),
                     '{"ratings":[,]}', '{"ratings":[1,,]}', '{"ratings":oops}', '{"ratings":[}']:
            with self.subTest(text=text), self.assertRaises(ResponseFormatError):
                decode_response(response(text))

    def test_truncation_and_refusal_reject_even_valid_json(self):
        for finish in ["length", "max_tokens", "content_filter"]:
            with self.subTest(finish=finish), self.assertRaises(ResponseFormatError):
                decode_response(response(json.dumps(PAYLOAD), finish))
        raw = response(json.dumps(PAYLOAD))
        raw['choices'][0]['message']['refusal'] = 'No'
        with self.assertRaises(ResponseFormatError):
            decode_response(raw)

    def test_http_non_json_has_safe_diagnostic(self):
        with tempfile.TemporaryDirectory() as cache:
            client = LlmClient(cache_dir=cache, api_key="SECRET", allow_live=True)
            http = Mock(ok=True, status_code=200)
            http.json.side_effect = ValueError('sensitive provider body')
            with patch('scenario_model.client.requests.post', return_value=http):
                with self.assertRaisesRegex(ResponseFormatError, r'API.*не JSON.*HTTP 200') as caught:
                    client._post({"model": "test"})
                self.assertNotIn('sensitive', str(caught.exception))

    def test_retry_cache_integrity_and_offline_replay(self):
        with tempfile.TemporaryDirectory() as cache:
            client = LlmClient(cache_dir=cache, api_key="SECRET", allow_live=True)
            body = {"model": "anthropic/claude-opus-4.6", "max_tokens": 16000}
            original = copy.deepcopy(body)
            def validate(payload):
                self.assertEqual(payload, PAYLOAD)
                return payload['ratings'][0], []
            with patch.object(client, '_post', side_effect=[response(''), response('```json\n' + json.dumps(PAYLOAD) + '\n```')]) as post, patch('scenario_model.client.time.sleep'):
                row, _, obs = client.observe(body, 'anthropic/claude-opus-4.6/common', validate)
                self.assertEqual(post.call_count, 2)
            self.assertEqual(body, original)
            self.assertEqual(obs['attempts'], 2)
            self.assertTrue(obs['transport_notes'])
            client.verify_observation(obs, body, obs['slot'])
            client.allow_live = False
            with patch.object(client, '_post', side_effect=AssertionError('network')):
                self.assertEqual(client.observe(body, obs['slot'], validate)[0], row)
            self.assertNotIn('SECRET', ''.join(p.read_text() for p in Path(cache).rglob('*.json')))

    def test_invalid_schema_is_retried_and_never_cached_as_success(self):
        with tempfile.TemporaryDirectory() as cache:
            client = LlmClient(cache_dir=cache, api_key="SECRET", allow_live=True)
            def reject(payload):
                raise ValueError('missing fields')
            with patch.object(client, '_post', return_value=response('{"ratings":[]}')), patch('scenario_model.client.time.sleep'):
                with self.assertRaises(ProviderError):
                    client.observe({'model': 'test'}, 'common', reject)
            self.assertEqual(len(list(Path(cache).glob('*/*.json'))), 0)
            self.assertEqual(len(list((Path(cache) / 'failed').rglob('*.json'))), 3)

    def test_empty_reply_explains_failure_after_three_attempts(self):
        with tempfile.TemporaryDirectory() as cache:
            client = LlmClient(cache_dir=cache, api_key="SECRET", allow_live=True)
            with patch.object(client, '_post', return_value=response('')) as post, patch('scenario_model.client.time.sleep'):
                with self.assertRaisesRegex(ProviderError, 'пустой ответ.*после 3 попыток'):
                    client.observe({'model': 'test'}, 'common', lambda x: (x, []))
            self.assertEqual(post.call_count, 3)


if __name__ == '__main__':
    unittest.main()
