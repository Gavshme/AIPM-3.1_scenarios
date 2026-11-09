# coding: utf-8
import copy, json, os, re, threading, time, uuid
from pathlib import Path
import requests
from .config import canonical, digest, DEFAULT_BASE_URL, verify_bundle


class MissingObservation(RuntimeError):
    pass


class ProviderError(RuntimeError):
    pass


_LOCKS = {}
_GUARD = threading.Lock()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(canonical(value), encoding="utf-8")
    tmp.replace(path)


class LlmClient:
    def __init__(
        self, cache_dir=".scenario_cache", api_key=None, base_url=None, allow_live=False
    ):
        self.cache = Path(cache_dir)
        self.api_key = (
            api_key if api_key is not None else os.environ.get("VSELLM_API_KEY", "")
        )
        self.base_url = (
            base_url or os.environ.get("VSELLM_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.allow_live = allow_live
        self.bundle_hash = verify_bundle()

    def fingerprint(self, body, slot):
        return digest(
            {
                "body": body,
                "slot": slot,
                "base_url": self.base_url,
                "bundle_hash": self.bundle_hash,
            }
        )

    def observe(self, body, slot, validator):
        fp = self.fingerprint(body, slot)
        path = self.cache / fp[:2] / (fp + ".json")
        with _GUARD:
            lock = _LOCKS.setdefault(str(path.resolve()), threading.Lock())
        with lock:
            if path.exists():
                obs = json.loads(path.read_text())
                self.verify_observation(obs, body, slot)
                row, notes = validator(obs["response"])
                return row, notes, obs
            if not self.allow_live:
                raise MissingObservation(
                    "Нет сохранённого этапа "
                    + slot
                    + ". Для новых оценок нажмите «Оценить сценарии»."
                )
            if not self.api_key:
                raise ProviderError("Не указан API-ключ.")
            for attempt in range(1, 4):
                raw = None
                try:
                    raw = self._post(body)
                    if raw["choices"][0].get("finish_reason") == "length":
                        raise ValueError("Ответ обрезан провайдером.")
                    text = raw["choices"][0]["message"]["content"]
                    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip()).strip()
                    payload = json.loads(text)
                    row, notes = validator(payload)
                    obs = {
                        "fingerprint": fp,
                        "slot": slot,
                        "base_url": self.base_url,
                        "bundle_hash": self.bundle_hash,
                        "request": copy.deepcopy(body),
                        "response": payload,
                        "response_hash": digest(payload),
                        "response_id": raw.get("id"),
                        "response_model": raw.get("model"),
                        "usage": raw.get("usage"),
                        "attempts": attempt,
                    }
                    atomic_json(path, obs)
                    return row, notes, obs
                except Exception as exc:
                    # Neither Authorization nor provider error bodies are written to disk/UI.
                    failure = {
                        "request": body,
                        "slot": slot,
                        "error_type": type(exc).__name__,
                        "attempt": attempt,
                    }
                    if isinstance(raw, dict):
                        failure["raw_response"] = {
                            k: raw[k]
                            for k in ["id", "model", "choices", "usage"]
                            if k in raw
                        }
                    atomic_json(
                        self.cache / "failed" / fp / (str(attempt) + ".json"), failure
                    )
                    if attempt == 3:
                        raise ProviderError(
                            "Не удалось завершить "
                            + slot
                            + " ("
                            + type(exc).__name__
                            + "). Успешные этапы сохранены; можно повторить запуск."
                        ) from None
                    time.sleep(2)

    def _post(self, body):
        response = requests.post(
            self.base_url + "/chat/completions",
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Cache-Control": "no-cache",
            },
            json=body,
            timeout=(20, 600),
        )
        if not response.ok:
            raise ProviderError("HTTP " + str(response.status_code))
        return response.json()

    def verify_observation(self, obs, body, slot):
        if (
            obs.get("bundle_hash") != self.bundle_hash
            or obs.get("base_url") != self.base_url
            or obs.get("request") != body
            or obs.get("slot") != slot
            or obs.get("fingerprint") != self.fingerprint(body, slot)
            or obs.get("response_hash") != digest(obs.get("response"))
        ):
            raise ValueError(
                "Сохранённый этап повреждён или относится к другому запросу."
            )
