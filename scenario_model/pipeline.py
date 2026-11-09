# coding: utf-8
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from .cards import normalize_card
from .config import MODELS, FEATURES, PROTOCOL_VERSION, RESULT_VERSION, digest
from .protocol import protocols, request_body, validate_stage, select_claim
from .scoring import ScenarioRidge
from .client import LlmClient


def _features(common, precision, benefits):
    return {
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


def evaluate_card(source, client, progress=None):
    card, ignored = normalize_card(source)
    p = protocols()
    scorer = ScenarioRidge()
    observations = []
    notes = []
    claim_rows = []

    def emit(text):
        if progress:
            progress(text)

    for repeat in range(3):
        emit("Выделение посыла: прочтение " + str(repeat + 1) + " из 3")
        spec = p["claim"]
        body = request_body(spec, card)
        row, nn, obs = client.observe(
            body,
            "claim/" + str(repeat + 1),
            lambda data: validate_stage("claim", spec, card, data),
        )
        claim_rows.append(row)
        observations.append(obs)
        notes.extend(nn)
    selection = select_claim([r["main_claim"] for r in claim_rows])
    claim = selection["fixed_main_claim"]
    emit("Посыл зафиксирован. Оценка семи признаков пятью моделями.")

    def assess(a):
        ratings = {}
        obs_all = []
        normalizations = []
        for stage in ["common", "precision", "benefits"]:
            spec = a["stages"][stage]
            body = request_body(spec, card, claim)
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
        "protocol_version": PROTOCOL_VERSION,
        "model_id": scorer.state["model_id"],
        "bundle_hash": scorer.bundle_hash,
        "input_card": copy.deepcopy(source),
        "card": card,
        "card_hash": digest(card),
        "ignored_fields": ignored,
        "claim": selection,
        "claim_evidence": [r["evidence"] for r in claim_rows],
        "assessors": assessed,
        "scoring": scoring,
        "observations": observations,
        "normalization_notes": notes,
    }


def replay_result(result):
    card, ignored = normalize_card(result["input_card"])
    scorer = ScenarioRidge()
    p = protocols()
    if (
        result.get("format") != RESULT_VERSION
        or result.get("protocol_version") != PROTOCOL_VERSION
        or result.get("bundle_hash") != scorer.bundle_hash
        or result.get("model_id") != scorer.state["model_id"]
        or result.get("card") != card
        or result.get("card_hash") != digest(card)
    ):
        raise ValueError("Несовместимый или изменённый результат.")
    observations = result.get("observations", [])
    if len(observations) != 18:
        raise ValueError("Нужны все 18 этапов результата.")
    byslot = {o["slot"]: o for o in observations}
    if len(byslot) != 18:
        raise ValueError("Повторяющиеся этапы результата.")

    def read(stage, spec, slot, claim=None):
        body = request_body(spec, card, claim)
        obs = byslot[slot]
        client = LlmClient(base_url=obs["base_url"], allow_live=False)
        client.verify_observation(obs, body, slot)
        row, _ = validate_stage(stage, spec, card, obs["response"], claim)
        return row

    claim_rows = [read("claim", p["claim"], "claim/" + str(i + 1)) for i in range(3)]
    selection = select_claim([r["main_claim"] for r in claim_rows])
    if selection != result["claim"]:
        raise ValueError("Посыл не совпадает с исходными прочтениями.")
    if result.get("claim_evidence") != [r["evidence"] for r in claim_rows]:
        raise ValueError("Обоснования посыла не совпадают с исходными ответами.")
    if result.get("ignored_fields") != ignored:
        raise ValueError("Список исключённых авторских полей изменён.")
    assessed = []
    for a in p["assessors"]:
        rr = {
            stage: read(
                stage,
                a["stages"][stage],
                a["model"] + "/" + stage,
                selection["fixed_main_claim"],
            )
            for stage in ["common", "precision", "benefits"]
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
