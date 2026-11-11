# coding: utf-8
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from .cards import normalize_card
from .config import (
    AUTHOR_PROTOCOL_VERSION, DEMO_FEATURE, DEMO_PROTOCOL_VERSION, MODELS,
    PROTOCOL_VERSION, RESULT_VERSION, ROOT, digest, read_json,
)
from .protocol import protocols, request_body, validate_stage, select_claim
from .scoring import ScenarioRidge
from .client import LlmClient


def _features(common, precision, benefits, function_demo=None):
    features = {
        "main_character": common["main_character"],
        "numeric_offer_on_screen": common["numeric_offer_on_screen"],
        "panel__distinct_benefit_count": min(4, len(benefits["benefit_groups"])),
        "panel__offer_novelty_explanation_need": common[
            "panel__offer_novelty_explanation_need"
        ],
        "promo": common["promo"],
        "panel__content_precision": precision["panel__content_precision"],
        "message_competition_mean": common["message_competition_mean"],
    }
    if function_demo is not None:
        features[DEMO_FEATURE] = function_demo["score"]
    return features


def _claim_plan(source, use_author_message, use_function_demo):
    if not use_author_message:
        spec = protocols()["claim"]
        return None, None, [("claim", spec, f"claim/{i}") for i in range(1, 4)]
    message = source.get("message")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Добавьте в сценарий раздел «Сообщение» с авторским посылом.")
    artifact = "function_demo.json" if use_function_demo else "author_message.json"
    bundle = read_json(ROOT / "artifacts" / artifact)
    return bundle, message.strip(), [("author_claim", bundle["claim"], "author_claim/1")]


def _claim_selection(rows, use_author_message):
    readings = [row["main_claim"] for row in rows]
    if not use_author_message:
        return select_claim(readings)
    return {
        "fixed_main_claim": readings[0],
        "readings": readings,
        "all_different": False,
        "source": "author_message",
    }


def _assessor_stages(assessor, demo_bundle):
    stages = {name: assessor["stages"][name] for name in ("common", "precision", "benefits")}
    if demo_bundle:
        stages["function_demo"] = {
            **demo_bundle["function_demo"],
            "model": assessor["model"],
            "parameters": stages["common"]["parameters"],
        }
    return stages


def evaluate_card(source, client, progress=None, *, use_author_message=False, use_function_demo=False):
    use_author_message = use_author_message or use_function_demo
    card, ignored = normalize_card(source)
    p = protocols()
    bundle, message, claim_plan = _claim_plan(source, use_author_message, use_function_demo)
    demo_bundle = bundle if use_function_demo else None
    scorer = ScenarioRidge(bundle)
    if use_author_message:
        ignored.remove("message")
    observations = []
    notes = []
    claim_rows = []

    def emit(text):
        if progress:
            progress(text)

    emit("Формулируется посыл сценария.")
    for stage, spec, slot in claim_plan:
        body = request_body(spec, card, message=message)
        row, nn, obs = client.observe(
            body,
            slot,
            lambda data: validate_stage(stage, spec, card, data, message=message),
        )
        claim_rows.append(row)
        observations.append(obs)
        notes.extend(nn)
    selection = _claim_selection(claim_rows, use_author_message)
    claim = selection["fixed_main_claim"]
    demo_target = None
    if demo_bundle:
        emit("Уточняется предложение для проверки показа функции.")
        spec = demo_bundle["demo_target"]
        row, nn, obs = client.observe(
            request_body(spec, card, claim),
            "demo_target/1",
            lambda data: validate_stage("demo_target", spec, card, data, claim),
        )
        demo_target = row["target"].strip()
        observations.append(obs)
        notes.extend(nn)
    emit(f"Посыл зафиксирован. Оценка {len(scorer.columns)} признаков пятью моделями.")

    def assess(a):
        ratings = {}
        obs_all = []
        normalizations = []
        for stage, spec in _assessor_stages(a, demo_bundle).items():
            body = request_body(spec, card, claim, demo_target=demo_target)
            row, nn, obs = client.observe(
                body,
                a["model"] + "/" + stage,
                lambda data: validate_stage(stage, spec, card, data, claim),
            )
            ratings[stage] = row
            obs_all.append(obs)
            normalizations.extend(nn)
        return {
            "model": a["model"],
            "features": _features(**ratings),
            "ratings": ratings,
            "observations": obs_all,
            "normalizations": normalizations,
        }

    assessed = [None] * len(p["assessors"])
    with ThreadPoolExecutor(max_workers=5) as pool:
        pending = {pool.submit(assess, a): i for i, a in enumerate(p["assessors"])}
        for count, future in enumerate(as_completed(pending), start=1):
            assessed[pending[future]] = future.result()
            emit(f"Готово оценщиков: {count} из 5.")
    for a in assessed:
        observations.extend(a.pop("observations"))
        notes.extend(a.pop("normalizations"))
    if [a["model"] for a in assessed] != MODELS:
        raise ValueError("Неверный состав оценщиков.")
    scoring = scorer.summarize([a["features"] for a in assessed])
    emit("Оценка завершена.")
    return {
        "format": RESULT_VERSION,
        "protocol_version": (
            DEMO_PROTOCOL_VERSION if use_function_demo
            else AUTHOR_PROTOCOL_VERSION if use_author_message
            else PROTOCOL_VERSION
        ),
        "model_id": scorer.state["model_id"],
        "bundle_hash": scorer.bundle_hash,
        "input_card": copy.deepcopy(source),
        "card": card,
        "card_hash": digest(card),
        "ignored_fields": ignored,
        "claim": selection,
        "claim_evidence": [r["evidence"] for r in claim_rows],
        **({"function_demo_target": demo_target} if demo_bundle else {}),
        "assessors": assessed,
        "scoring": scoring,
        "observations": observations,
        "normalization_notes": notes,
    }


def explain_score(result):
    version = result["protocol_version"]
    bundle, _, _ = _claim_plan(
        result["input_card"],
        version in {AUTHOR_PROTOCOL_VERSION, DEMO_PROTOCOL_VERSION},
        version == DEMO_PROTOCOL_VERSION,
    )
    return ScenarioRidge(bundle).feature_contributions(result["scoring"]["model_inputs"])


def replay_result(result):
    source = result["input_card"]
    card, ignored = normalize_card(source)
    version = result.get("protocol_version")
    use_function_demo = version == DEMO_PROTOCOL_VERSION
    use_author_message = version in {AUTHOR_PROTOCOL_VERSION, DEMO_PROTOCOL_VERSION}
    bundle, message, claim_plan = _claim_plan(source, use_author_message, use_function_demo)
    demo_bundle = bundle if use_function_demo else None
    scorer = ScenarioRidge(bundle)
    if use_author_message:
        ignored.remove("message")
    p = protocols()
    if (
        result.get("format") != RESULT_VERSION
        or version not in {PROTOCOL_VERSION, AUTHOR_PROTOCOL_VERSION, DEMO_PROTOCOL_VERSION}
        or result.get("bundle_hash") != scorer.bundle_hash
        or result.get("model_id") != scorer.state["model_id"]
        or result.get("card") != card
        or result.get("card_hash") != digest(card)
    ):
        raise ValueError("Несовместимый или изменённый результат.")
    observations = result.get("observations", [])
    expected_count = (
        len(claim_plan) + int(use_function_demo)
        + sum(len(_assessor_stages(a, demo_bundle)) for a in p["assessors"])
    )
    if len(observations) != expected_count:
        raise ValueError(f"Нужны все {expected_count} этапов результата.")
    byslot = {o["slot"]: o for o in observations}
    if len(byslot) != expected_count:
        raise ValueError("Повторяющиеся этапы результата.")

    def read(stage, spec, slot, claim=None, *, message=None, demo_target=None):
        body = request_body(spec, card, claim, message=message, demo_target=demo_target)
        obs = byslot[slot]
        client = LlmClient(base_url=obs["base_url"], allow_live=False)
        client.verify_observation(obs, body, slot)
        row, _ = validate_stage(stage, spec, card, obs["response"], claim, message=message)
        return row

    claim_rows = [read(stage, spec, slot, message=message) for stage, spec, slot in claim_plan]
    selection = _claim_selection(claim_rows, use_author_message)
    if selection != result["claim"]:
        raise ValueError("Посыл не совпадает с исходными прочтениями.")
    if result.get("claim_evidence") != [r["evidence"] for r in claim_rows]:
        raise ValueError("Обоснования посыла не совпадают с исходными ответами.")
    if result.get("ignored_fields") != ignored:
        raise ValueError("Список исключённых авторских полей изменён.")
    demo_target = None
    if demo_bundle:
        demo_target = read(
            "demo_target", demo_bundle["demo_target"], "demo_target/1",
            selection["fixed_main_claim"],
        )["target"].strip()
        if demo_target != result.get("function_demo_target"):
            raise ValueError("Изменено предложение для оценки показа функции.")
    assessed = []
    for a in p["assessors"]:
        rr = {
            stage: read(
                stage,
                spec,
                a["model"] + "/" + stage,
                selection["fixed_main_claim"],
                demo_target=demo_target,
            )
            for stage, spec in _assessor_stages(a, demo_bundle).items()
        }
        assessed.append(
            {"model": a["model"], "features": _features(**rr), "ratings": rr}
        )
    if assessed != result["assessors"]:
        raise ValueError("Оценки не совпадают с исходными ответами.")
    scoring = scorer.summarize([a["features"] for a in assessed])
    stored = result["scoring"]
    if scoring["class"] != stored["class"]:
        raise ValueError("Класс не совпадает с пересчётом.")
    for k in ["score", "assessor_scores", "score_range"]:
        if not np.allclose(scoring[k], stored[k], rtol=0, atol=1e-10):
            raise ValueError("Скор не совпадает с пересчётом.")
    if (
        scoring["model_inputs"] != stored["model_inputs"]
        or scoring["feature_means"] != stored["feature_means"]
        or scoring["feature_ranges"] != stored["feature_ranges"]
    ):
        raise ValueError("Признаки не совпадают с пересчётом.")
    return scoring
