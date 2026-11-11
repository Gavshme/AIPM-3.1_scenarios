# coding: utf-8
import json, os
from html import escape
from jsonschema.exceptions import ValidationError
import streamlit as st
from scenario_model.cards import parse_cards
from scenario_model.client import LlmClient
from scenario_model.config import DEFAULT_BASE_URL, LABELS, ROOT, digest
from scenario_model.documents import parse_upload
from scenario_model.explanations import SUMMARY_MODEL, add_summary, read_summary, short_saved_reason
from scenario_model.pipeline import evaluate_card, explain_score, replay_result


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
    "AI формулирует посыл из раздела «Сообщение», затем пять оценщиков измеряют восемь признаков."
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


CLASS_COLORS = {
    "Высокий": ("#eaf7ee", "#166534", "#b0dbc0"),
    "Средний": ("#fff5d5", "#804700", "#efd28b"),
    "Низкий": ("#fff0ef", "#9c2424", "#edb4b0"),
}


def result_cards(result):
    scoring = result["scoring"]
    label = scoring["class"]
    background, color, border = CLASS_COLORS[label]
    st.html(f"""
        <style>
            .aipm-result-cards {{ display: grid; grid-template-columns: minmax(8rem, 1fr) 3fr; gap: 1rem; }}
            .aipm-result-card {{ border: 1px solid; border-radius: 10px; padding: 1rem; overflow-wrap: anywhere; }}
            .aipm-result-card strong {{ display: block; margin-bottom: .35rem; }}
            @media (max-width: 600px) {{ .aipm-result-cards {{ grid-template-columns: 1fr; gap: .5rem; }} }}
        </style>
        <div class="aipm-result-cards">
            <div class="aipm-result-card" style="background:{background};color:{color};border-color:{border}">
                <strong>Скор MD</strong>
                <div style="font-size:2rem;font-weight:700;line-height:1.15">{scoring["score"]:.1f}</div>
                <div>{label} класс</div>
            </div>
            <div class="aipm-result-card" style="background:#f6f8fa;color:#202936;border-color:#dce1e7">
                <strong>Основной посыл</strong>
                <div>{escape(result["claim"]["fixed_main_claim"])}</div>
            </div>
        </div>
    """)


def render_results(results):
    if not results:
        return
    st.subheader("Результаты оценки")
    for result in sorted(results, key=lambda r: -r["scoring"]["score"]):
        with st.expander(
            str(result["input_card"].get("name", result["card"]["id"])),
            expanded=True,
        ):
            result_cards(result)
            baseline, contributions = explain_score(result)
            means = result["scoring"]["feature_means"]
            rows = []
            for feature, value in means.items():
                delta = round(contributions[feature], 2)
                rows.append({
                    "Признак": LABELS[feature],
                    "Оценка": f"{value:.2f}".rstrip("0").rstrip("."),
                    "Вклад в MD": f"{delta:+.2f}".replace("-", "−") if delta else "0",
                })
            st.dataframe(rows, hide_index=True, width="stretch")
            st.caption(
                f"Вклад в пунктах MD относительно базового скора {baseline:.2f}. "
                "Плюс повышает итоговый скор, минус снижает."
            )
            try:
                summary = read_summary(result)
            except (ValidationError, ValueError, TypeError, KeyError):
                summary = {}
            st.markdown("**Почему признаки получили такие оценки**")
            if not summary:
                st.caption("Сводка AI недоступна. Показано по одному сохранённому обоснованию на признак.")
            for feature in means:
                row = summary.get(feature)
                conclusion = row["conclusion"] if row else short_saved_reason(result, feature)
                st.markdown(f"**{LABELS[feature]}**")
                st.text(conclusion or "Обоснование отсутствует.")
                if row and row["quote"]:
                    st.caption("В сценарии: «" + row["quote"] + "»")


new_tab, saved_tab = st.tabs(["Оценить сценарии", "Открыть результат"])
with new_tab:
    st.write(
        "Загрузите полные сценарии в DOCX или PDF с таблицей сцен и разделом «Сообщение»."
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
        st.caption(f"Карточек: {len(cards)}.")
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
                        result = evaluate_card(
                            item["original"],
                            client,
                            progress=lambda text: status.update(label=text),
                            use_function_demo=True,
                        )
                        status.update(label="Готовится общее объяснение по признакам.")
                        try:
                            add_summary(result, client, setting("SCENARIO_SUMMARY_MODEL", SUMMARY_MODEL))
                        except (ValidationError, RuntimeError, ValueError, TypeError, KeyError, OSError):
                            pass
                        results.append(result)
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
