# coding: utf-8
from pathlib import Path
import hashlib, json

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "main_character",
    "numeric_offer_on_screen",
    "panel__distinct_benefit_count",
    "panel__offer_novelty_explanation_need",
    "promo",
    "panel__content_precision",
    "message_competition_mean",
]
DEMO_FEATURE = "panel__key_service_function_demonstration"
DEMO_FEATURES = [*FEATURES, DEMO_FEATURE]
LABELS = dict(
    zip(
        FEATURES,
        [
            "Главный персонаж",
            "Числовое предложение",
            "Количество выгод",
            "Необходимость объяснения новизны",
            "Промо",
            "Точность содержания",
            "Конкуренция посылов",
        ],
    )
)
LABELS[DEMO_FEATURE] = "Показ ключевой функции сервиса/продукта"
MODELS = [
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-sonnet-4.6",
    "openai/gpt-5.4",
    "x-ai/grok-4.6",
    "anthropic/claude-opus-4.6",
]
ACTIVE_MODELS = [model for model in MODELS if model != "x-ai/grok-4.6"]
PROTOCOL_VERSION = "scenario-blind-claim-five-v1"
AUTHOR_PROTOCOL_VERSION = "scenario-author-message-five-v1"
DEMO_PROTOCOL_VERSION = "scenario-author-message-function-demo-five-v1"
FULL_CLAIM_PROTOCOL_VERSION = "scenario-author-message-function-demo-full-claim-five-v1"
FOUR_ASSESSOR_PROTOCOL_VERSION = "scenario-author-message-function-demo-full-claim-four-v1"
ORDINAL_CLASS_PROTOCOL_VERSION = "scenario-author-message-function-demo-full-claim-four-ordinal-v1"
FOUR_ASSESSOR_PROTOCOLS = {FOUR_ASSESSOR_PROTOCOL_VERSION, ORDINAL_CLASS_PROTOCOL_VERSION}
FUNCTION_DEMO_PROTOCOLS = {
    DEMO_PROTOCOL_VERSION, FULL_CLAIM_PROTOCOL_VERSION, *FOUR_ASSESSOR_PROTOCOLS,
}
RESULT_VERSION = "scenario-result-v1"
DEFAULT_BASE_URL = "https://litellm.data-light.ru/v1"


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_bundle():
    manifest = read_json(ROOT / "artifacts/manifest.json")
    for name, expected in manifest["files_sha256"].items():
        path = (ROOT / name).resolve()
        if (
            ROOT not in path.parents
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("Нарушена целостность модели или протокола: " + name)
    return digest(manifest)
