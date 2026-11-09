# coding: utf-8
import json, os
from pathlib import Path
import streamlit as st
from scenario_model.cards import parse_cards
from scenario_model.client import LlmClient
from scenario_model.config import DEFAULT_BASE_URL, FEATURES, LABELS, ROOT, digest
from scenario_model.documents import parse_upload
from scenario_model.pipeline import evaluate_card, replay_result


def setting(name, default=""):
    value = os.environ.get(name)
    if value:
        return value
    try:
        return st.secrets.get(name, default)
    except (FileNotFoundError, KeyError):
        return default


st.set_page_config(page_title="AIPM 3.1 · Сценарии", page_icon="📝", layout="wide")
st.title("Считываемость рекламных сценариев")
st.write(
    "Сравните, насколько понятно сценарии передают предложение сервиса и пользу для пользователя."
)
st.caption(
    "AI автоматически выделяет посыл из сцен, реплик и экранного текста, затем пять оценщиков измеряют семь признаков."
)
key = setting("VSELLM_API_KEY")
base_url = setting("VSELLM_BASE_URL", DEFAULT_BASE_URL)
if not key:
    with st.sidebar:
        st.subheader("Подключение")
        key = st.text_input(
            "API-ключ",
            type="password",
            help="Для новой оценки. Можно задать VSELLM_API_KEY в окружении.",
        )
        st.caption("Для открытия сохранённого результата ключ не нужен.")
        with st.expander("Настройки подключения"):
            base_url = st.text_input("Адрес API", value=base_url)


def render_results(results):
    if not results:
        return
    st.subheader("Сравнение сценариев")
    st.dataframe(
        [
            {
                "Сценарий": r["input_card"].get("name", r["card"]["id"]),
                "Скор MD": round(r["scoring"]["score"], 2),
                "Класс": r["scoring"]["class"],
            }
            for r in sorted(results, key=lambda r: -r["scoring"]["score"])
        ],
        hide_index=True,
        width="stretch",
    )
    for r in results:
        with st.expander(
            str(r["input_card"].get("name", r["card"]["id"])),
            expanded=len(results) == 1,
        ):
            a, b = st.columns([1, 3])
            a.metric("Скор MD", f"{r['scoring']['score']:.1f}")
            a.write(r["scoring"]["class"])
            b.markdown("**Выделенный посыл**")
            b.write(r["claim"]["fixed_main_claim"])
            if r["claim"]["all_different"]:
                st.warning(
                    "Три прочтения дали разные формулировки посыла. По правилу выбран первый вариант; проверьте неоднозначные места сценария."
                )
            st.dataframe(
                [
                    {
                        "Признак": LABELS[c],
                        "Средняя оценка": round(r["scoring"]["feature_means"][c], 2),
                        "Минимум": r["scoring"]["feature_ranges"][c][0],
                        "Максимум": r["scoring"]["feature_ranges"][c][1],
                    }
                    for c in FEATURES
                ],
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Минимум и максимум — исходные оценки пяти AI-моделей, не доверительный интервал ответов зрителей."
            )
            with st.expander("Как был определён посыл"):
                for i, (text, evidence) in enumerate(
                    zip(r["claim"]["readings"], r["claim_evidence"])
                ):
                    st.markdown(f"**Прочтение {i+1}**")
                    st.write(text)
                    st.caption(evidence)
            with st.expander("Обоснования оценок"):
                for assessor in r["assessors"]:
                    st.markdown("### " + assessor["model"].split("/")[-1])
                    ratings = assessor["ratings"]
                    for c in FEATURES:
                        st.markdown(
                            "**"
                            + LABELS[c]
                            + " · "
                            + str(assessor["features"][c])
                            + "**"
                        )
                        if c == "panel__distinct_benefit_count":
                            st.write(ratings["benefits"]["assessment_reason"])
                            for candidate in ratings["benefits"]["candidates"]:
                                st.write("«" + candidate["quote"] + "»")
                                st.caption(candidate["decision_reason"])
                        elif c == "panel__content_precision":
                            st.write(ratings["precision"]["reason"])
                            st.caption(ratings["precision"]["evidence"])
                        else:
                            st.write(ratings["common"]["evidence"][c])


new_tab, saved_tab = st.tabs(["Оценить сценарии", "Открыть результат"])
with new_tab:
    st.write(
        "Загрузите полные сценарии в DOCX или PDF с таблицей сцен. Посыл определится автоматически."
    )
    files = st.file_uploader(
        "Файлы сценариев",
        type=["docx", "pdf", "json"],
        accept_multiple_files=True,
        key="cards_upload",
    )
    with st.expander("Дополнительно: формат JSON"):
        pasted = st.text_area("Текст карточек", height=180, key="cards_text")
        example = (ROOT / "examples/scenario.json").read_text(encoding="utf-8")
        st.download_button(
            "Скачать пример карточки",
            example,
            file_name="scenario.json",
            mime="application/json",
        )
    cards = []
    error = None
    try:
        for f in files or []:
            cards.extend(parse_upload(f.name, f.getvalue()))
        if pasted.strip():
            cards.extend(parse_cards(pasted))
        if len({x["card"]["id"] for x in cards}) != len(cards):
            raise ValueError("В загрузке повторяются id карточек.")
        if len(cards) > 100:
            raise ValueError("За один раз можно оценить не больше 100 карточек.")
    except Exception as exc:
        error = str(exc)
        st.error(error)
    fingerprint = digest([x["original"] for x in cards])
    if st.session_state.get("input_fingerprint") != fingerprint:
        st.session_state["input_fingerprint"] = fingerprint
        st.session_state["evaluated"] = []
        st.session_state["evaluation_errors"] = []
    if cards:
        st.caption(
            f"Карточек: {len(cards)}. Авторские разделы «Сообщение» и «Ключевая идея» не используются для выделения посыла."
        )
    c1, c2 = st.columns(2)
    live = c1.button(
        "Оценить сценарии", type="primary", disabled=not cards or bool(error)
    )
    cached = c2.button(
        "Рассчитать из сохранённых ответов", disabled=not cards or bool(error)
    )
    if live or cached:
        client = LlmClient(
            api_key=key,
            base_url=base_url,
            allow_live=live,
            cache_dir=os.environ.get(
                "SCENARIO_CACHE_DIR", str(ROOT / ".scenario_cache")
            ),
        )
        results = []
        errors = []
        if live and not client.api_key:
            st.error("Укажите API-ключ в боковой панели или окружении.")
        else:
            for item in cards:
                with st.status("Оценивается " + item["name"], expanded=False) as status:
                    try:
                        results.append(
                            evaluate_card(
                                item["original"],
                                client,
                                progress=lambda text: status.update(label=text),
                            )
                        )
                        status.update(
                            label=item["name"] + " — готово", state="complete"
                        )
                    except Exception as exc:
                        errors.append({"id": item["card"]["id"], "error": str(exc)})
                        status.update(
                            label=item["name"] + " — оценка не завершена", state="error"
                        )
                        st.error(str(exc))
            st.session_state["evaluated"] = results
            st.session_state["evaluation_errors"] = errors
    render_results(st.session_state.get("evaluated", []))
    if st.session_state.get("evaluated"):
        export = {
            "format": "scenario-batch-v1",
            "results": st.session_state["evaluated"],
            "errors": st.session_state.get("evaluation_errors", []),
        }
        st.download_button(
            "Скачать оценки и обоснования",
            json.dumps(export, ensure_ascii=False, indent=2),
            file_name="scenario_results.json",
            mime="application/json",
        )
    if st.session_state.get("evaluation_errors"):
        st.info(
            "Незавершённые карточки не получили скор. Повторный запуск использует уже сохранённые успешные этапы."
        )
with saved_tab:
    saved = st.file_uploader(
        "Сохранённый результат оценки", type=["json"], key="result_upload"
    )
    if saved:
        try:
            data = json.loads(saved.getvalue().decode("utf-8-sig"))
            results = data.get("results", [data])
            if not results:
                raise ValueError("Нет завершённых результатов.")
            for r in results:
                replay_result(r)
            st.success("Сохранённые оценки проверены без обращения к API.")
            render_results(results)
        except Exception as exc:
            st.error("Не удалось проверить результат: " + str(exc))
