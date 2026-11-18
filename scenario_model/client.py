# coding: utf-8
import copy, json, os, re, threading, time, uuid
from pathlib import Path
import requests
from .config import canonical, digest, DEFAULT_BASE_URL, verify_bundle


class MissingObservation(RuntimeError):
    pass


class ProviderError(RuntimeError):
    pass


class ResponseFormatError(ProviderError):
    """A safe diagnostic, without provider bodies or scenario text."""


def _json_document(text):
    """Isolate one complete JSON root, never recover a nested partial result."""
    start = re.search(r"[\[{]", text)
    if start is None:
        raise ResponseFormatError("Модель вернула текст без JSON с оценками.")
    stack, quoted, escaped = [], False, False
    for i in range(start.start(), len(text)):
        char = text[i]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            if not stack or stack.pop() != (
                "[" if char == "]" else "{"
            ):
                raise ResponseFormatError("В JSON модели нарушены скобки.")
            if not stack:
                if re.search(r"[\[{]", text[i + 1:]):
                    raise ResponseFormatError("Модель вернула несколько JSON-блоков вместо одного.")
                return text[start.start():i + 1]
    raise ResponseFormatError("JSON модели не завершён: ответ обрезан или повреждён.")


def _remove_trailing_commas(text):
    """Remove only commas before closing brackets, preserving all string data."""
    result, quoted, escaped = [], False, False
    for i, char in enumerate(text):
        if quoted:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
            result.append(char)
        elif char == "," and re.match(r"\s*[\]}]", text[i + 1:]):
            previous = "".join(result).rstrip()
            if not previous or previous[-1] in "[{,:":
                result.append(char)
        else:
            result.append(char)
    return "".join(result)


def decode_response(raw):
    try:
        choice = raw["choices"][0]
        message = choice["message"]
        content = message.get("content")
    except (KeyError, IndexError, TypeError, AttributeError):
        raise ResponseFormatError("API вернул ответ без choices/message.") from None
    if choice.get("finish_reason") in {"length", "max_tokens"}:
        raise ResponseFormatError("Ответ модели обрезан лимитом токенов.")
    if message.get("refusal") or choice.get("finish_reason") == "content_filter":
        raise ResponseFormatError("Модель отказалась возвращать оценки.")
    if isinstance(content, list):
        if not all(isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str) for p in content):
            raise ResponseFormatError("API вернул неподдерживаемый формат текстовых блоков.")
        content = "".join(p["text"] for p in content)
    if content is None or isinstance(content, str) and not content.strip():
        raise ResponseFormatError("Модель вернула пустой ответ вместо оценок.")
    if not isinstance(content, str):
        raise ResponseFormatError("API вернул нетекстовый ответ модели.")
    document = _json_document(content.lstrip("\ufeff").strip())
    cleaned = _remove_trailing_commas(document)
    literal_control_characters = False
    try:
        # Literal newlines inside evidence strings are safe to preserve. No
        # missing fields, quotes, brackets or feature values are invented.
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            payload = json.loads(cleaned, strict=False)
            literal_control_characters = True
    except json.JSONDecodeError as exc:
        raise ResponseFormatError(
            f"В JSON модели ошибка синтаксиса (строка {exc.lineno}, столбец {exc.colno})."
        ) from None
    notes = []
    if document != content.strip() or cleaned != document:
        notes.append("Убрано внешнее оформление JSON или завершающие запятые; оценки не изменены.")
    if literal_control_characters:
        notes.append("Сохранены буквальные переносы строк/управляющие символы внутри текстовых обоснований.")
    return payload, notes


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
                    payload, transport_notes = decode_response(raw)
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
                        "transport_notes": transport_notes,
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
                    if isinstance(exc, ResponseFormatError):
                        failure["error_detail"] = str(exc)
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
                        detail = str(exc) if isinstance(exc, ResponseFormatError) else type(exc).__name__
                        raise ProviderError(
                            "Не удалось завершить "
                            + slot
                            + " ("
                            + detail
                            + ") после 3 попыток. Успешные этапы сохранены; можно повторить запуск."
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
        try:
            return response.json()
        except ValueError:
            raise ResponseFormatError(
                "API вернул не JSON (HTTP " + str(response.status_code) + ")."
            ) from None

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
