# coding: utf-8
import copy, json, math, re
from .config import digest

CONTEXT = ["voice_and_music", "sound_design", "creative_techniques"]


def normalize_card(source):
    if not isinstance(source, dict):
        raise ValueError("Карточка должна быть объектом JSON.")
    card = {
        k: copy.deepcopy(source[k])
        for k in ["id", "duration_sec", "scenes"]
        if k in source
    }
    if not isinstance(card.get("id"), str) or not card["id"].strip():
        raise ValueError("У карточки отсутствует строковый id.")
    if len(card["id"]) > 128:
        raise ValueError("id длиннее 128 символов.")
    duration = card.get("duration_sec")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or not 0 < duration <= 3600
    ):
        raise ValueError(
            "duration_sec должен быть положительным числом секунд, не больше 3600."
        )
    if (
        not isinstance(card.get("scenes"), list)
        or not card["scenes"]
        or len(card["scenes"]) > 200
    ):
        raise ValueError("Нужен список scenes: от 1 до 200 сцен.")
    scenes = []
    previous_end = 0.0
    for i, s in enumerate(card["scenes"]):
        if not isinstance(s, dict):
            raise ValueError("Сцена должна быть объектом.")
        text = {k: s.get(k, "") for k in ["visual", "speech", "screen_text"]}
        if not all(isinstance(v, str) for v in text.values()):
            raise ValueError("visual, speech и screen_text должны быть строками.")
        if isinstance(s.get("time"), str):
            nums = re.findall(r"\d+(?:[.,]\d+)?", s["time"])
            if len(nums) != 2:
                raise ValueError(
                    "Укажите time как интервал секунд, например «0,0–5,0 с»."
                )
            start, end = [float(n.replace(",", ".")) for n in nums]
            time_text = s["time"]
        elif all(
            isinstance(s.get(k), (int, float)) and not isinstance(s[k], bool)
            for k in ["start", "end"]
        ):
            start, end = float(s["start"]), float(s["end"])
            time_text = f"{start:g}–{end:g} с"
        else:
            raise ValueError("Для каждой сцены нужен time или start/end в секундах.")
        if (
            not all(math.isfinite(v) for v in [start, end])
            or not 0 <= start < end <= duration + 0.11
            or start < previous_end - 0.11
        ):
            raise ValueError(
                f"Проверьте интервал сцены {i+1}: он выходит за длительность или перекрывает предыдущий."
            )
        previous_end = end
        scenes.append({"time": time_text, **text})
    if not any(
        s[k].strip() for s in scenes for k in ["visual", "speech", "screen_text"]
    ):
        raise ValueError(
            "В сценах нет описания изображения, реплик или экранного текста."
        )
    card["scenes"] = scenes
    for key in CONTEXT:
        if key in source:
            if not isinstance(source[key], (str, list, dict)):
                raise ValueError(key + ": ожидается текст, список или объект.")
            card[key] = copy.deepcopy(source[key])
    ignored = sorted(set(source) - set(card) - {"name"})
    return card, ignored


def load_cards(value):
    if isinstance(value, dict) and "cards" in value:
        value = value["cards"]
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list) or not 1 <= len(value) <= 100:
        raise ValueError("Загрузите одну карточку или список из 1–100 карточек.")
    cards = []
    ids = set()
    for item in value:
        card, ignored = normalize_card(item)
        if card["id"] in ids:
            raise ValueError("Повторяющийся id: " + card["id"])
        ids.add(card["id"])
        cards.append(
            {
                "card": card,
                "original": copy.deepcopy(item),
                "ignored_fields": ignored,
                "name": str(item.get("name", card["id"])),
            }
        )
    return cards


def parse_cards(text):
    if len(text.encode("utf-8")) > 5_000_000:
        raise ValueError("JSON больше 5 МБ.")
    return load_cards(json.loads(text))


def claim_input(card):
    # The historical claim stage used start/end, without audio/creative or author fields.
    scenes = []
    for s in card["scenes"]:
        start, end = [
            float(x.replace(",", "."))
            for x in re.findall(r"\d+(?:[.,]\d+)?", s["time"])
        ]
        scenes.append(
            {
                "start": start,
                "end": end,
                **{k: s[k] for k in ["visual", "screen_text", "speech"]},
            }
        )
    return {"id": card["id"], "duration_sec": card["duration_sec"], "scenes": scenes}


def feature_input(card, claim):
    return {
        "id": card["id"],
        "fixed_main_claim": claim,
        **{k: copy.deepcopy(v) for k, v in card.items() if k != "id"},
    }
