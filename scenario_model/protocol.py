# coding: utf-8
import copy, re
from jsonschema import validate
from .config import ROOT, read_json, verify_bundle
from .cards import claim_input, feature_input


class ProtocolError(ValueError):
    pass


def protocols():
    verify_bundle()
    return read_json(ROOT / "protocols/stages.json")


def request_body(spec, card, claim=None):
    data = claim_input(card) if claim is None else feature_input(card, claim)
    return {
        "model": spec["model"],
        **copy.deepcopy(spec["parameters"]),
        "response_format": copy.deepcopy(spec["response_format"]),
        "messages": [
            {
                "role": "user",
                "content": spec["prompt"]
                + spec["marker"]
                + __import__("json").dumps([data], ensure_ascii=False, sort_keys=True),
            }
        ],
    }


def canon(s):
    return s.strip().replace("ё", "е").replace("Ё", "Е")


def tokens(s):
    return re.findall(r"\w+", s.lower().replace("ё", "е"))


def grounded(quote, text):
    a, b = tokens(quote), tokens(text)
    return bool(a) and any(b[i : i + len(a)] == a for i in range(len(b) - len(a) + 1))


def validate_stage(stage, spec, card, payload, claim=None):
    payload = copy.deepcopy(payload)
    notes = []
    if isinstance(payload, list):
        payload = {"ratings": payload}
        notes.append("Ответ-массив приведён к ratings.")
    schema = copy.deepcopy(spec["response_format"]["json_schema"]["schema"])
    if stage == "common":
        e = schema["properties"]["ratings"]["items"]["properties"]["evidence"]
        e["required"] = [k for k in e["required"] if k != "panel__content_precision"]
        for row in payload.get("ratings", []):
            for k in list(row.get("evidence", {})):
                if k not in e["properties"]:
                    row["evidence"].pop(k)
                    notes.append("Удалено неиспользуемое поле evidence." + k)
    validate(payload, schema)
    if len(payload["ratings"]) != 1 or payload["ratings"][0]["id"] != card["id"]:
        raise ProtocolError("Неверный id или неполный ответ.")
    row = payload["ratings"][0]
    if stage == "claim":
        if not row["main_claim"].strip() or not row["evidence"].strip():
            raise ProtocolError("Нет посыла или обоснования его выделения.")
        if row["panel__distinct_benefit_count"] != min(4, len(row["benefits"])):
            raise ProtocolError("Служебное число выгод не согласовано со списком.")
    elif stage in ["common", "precision"]:
        if canon(row["main_claim"]) != canon(claim):
            raise ProtocolError("Оценщик изменил зафиксированный посыл.")
        if row["main_claim"] != claim:
            notes.append("Пробелы/е-ё в посыле приведены к исходной формулировке.")
            row["main_claim"] = claim
        if stage == "common":
            for k in [
                "main_character",
                "numeric_offer_on_screen",
                "panel__offer_novelty_explanation_need",
                "promo",
                "message_competition_mean",
            ]:
                if not row["evidence"][k].strip():
                    raise ProtocolError("Пустое обоснование: " + k)
            for k, value in [
                ("main_character", int(row["main_character_status"] == "confirmed")),
                ("message_competition_mean", max(len(row["messages"]) - 1, 0)),
            ]:
                if row[k] != value:
                    notes.append(
                        k
                        + ": значение "
                        + str(row[k])
                        + " приведено к "
                        + str(value)
                        + " по исходному правилу."
                    )
                    row[k] = value
        elif not row["reason"].strip() or not row["evidence"].strip():
            raise ProtocolError("Нет обоснования точности содержания.")
    elif stage == "benefits":
        if not row["assessment_reason"].strip():
            raise ProtocolError("Нет обоснования количества выгод.")
        ids = [g["group_id"] for g in row["benefit_groups"]]
        if len(ids) != len(set(ids)):
            raise ProtocolError("Повторяющиеся номера групп выгод.")
        allowed = set(ids)
        supported = set()
        for g in row["benefit_groups"]:
            if not g["benefit"].strip() or not g["why_distinct"].strip():
                raise ProtocolError("Нет описания самостоятельной выгоды.")
        for c in row["candidates"]:
            m = re.fullmatch(
                r"scenes\[(\d+)\]\.(visual|speech|screen_text)", c["source_path"]
            )
            if not m or int(m[1]) >= len(card["scenes"]):
                raise ProtocolError("Неверная ссылка на сцену.")
            if not grounded(c["quote"], card["scenes"][int(m[1])][m[2]]):
                raise ProtocolError("Цитата не найдена в указанном поле сцены.")
            if not c["claim"].strip() or not c["decision_reason"].strip():
                raise ProtocolError("Нет обоснования кандидата в выгоды.")
            gid = c["group_id"]
            if gid != 0 and gid not in allowed:
                raise ProtocolError("Кандидат ссылается на неизвестную группу.")
            if gid:
                supported.add(gid)
        if supported != allowed:
            raise ProtocolError("Группа выгод не подтверждена цитатой.")
    return row, notes


def select_claim(readings):
    if len(readings) != 3 or any(
        not isinstance(r, str) or not r.strip() for r in readings
    ):
        raise ProtocolError("Нужны три содержательных прочтения посыла.")
    chosen = max(readings, key=lambda x: readings.count(x))
    return {
        "fixed_main_claim": chosen,
        "readings": readings,
        "exact_agreement": readings.count(chosen),
        "all_different": len(set(readings)) == 3,
        "selection_rule": "Наиболее частая дословная формулировка; при равенстве — первый повтор.",
    }
