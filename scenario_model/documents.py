"""Read the scenario table and audiovisual sections from uploaded documents."""

import hashlib
from io import BytesIO
from pathlib import Path
import re
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from .cards import load_cards, parse_cards

WORD = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
CONTEXT_HEADINGS = {
    "голос и музыка": "voice_and_music",
    "звуковой дизайн": "sound_design",
    "креативные приемы": "creative_techniques",
}
NUMBER = r"\d+(?:[.,]\d+)?"
TIMECODE = rf"(?:\d+:){{0,2}}{NUMBER}"
INTERVAL = re.compile(rf"\s*({TIMECODE})\s*[–—−-]\s*({TIMECODE})\s*(?:с\.?|сек\.?)?\s*")
SCREEN_LABEL = re.compile(r"(?im)^\s*(?:экранный текст|текст на экране|супер)\s*:\s*")
FINAL_SCREEN = re.compile(r"экранный текст на финальном экране\s*:\s*(.+)", re.I)
DURATION = re.compile(
    rf"(?:хронометраж|длительность)\s*:\s*({TIMECODE})\s*(?:с\.?|сек\.?)?", re.I
)


def _seconds(value):
    seconds = 0.0
    for part in value.replace(",", ".").split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def _paragraph(element):
    pieces = []
    for node in element.iter():
        if node.tag == WORD + "t":
            pieces.append(node.text or "")
        elif node.tag in {WORD + "br", WORD + "cr", WORD + "tab"}:
            pieces.append("\n" if node.tag != WORD + "tab" else "\t")
    return "".join(pieces).strip()


def _word_blocks(data):
    with ZipFile(BytesIO(data)) as archive:
        if archive.getinfo("word/document.xml").file_size > 10_000_000:
            raise ValueError("В DOCX слишком много текста.")
        document = ElementTree.fromstring(archive.read("word/document.xml"))
        body = document.find(WORD + "body")
        if body is None:
            raise ValueError("В DOCX нет основного текста.")
    for element in body:
        if element.tag == WORD + "p":
            yield _paragraph(element)
        elif element.tag == WORD + "tbl":
            yield [
                [
                    "\n".join(filter(None, map(_paragraph, cell.findall(WORD + "p"))))
                    for cell in row.findall(WORD + "tc")
                ]
                for row in element.findall(WORD + "tr")
            ]


def _pdf_blocks(data):
    import pdfplumber

    with pdfplumber.open(BytesIO(data)) as pdf:
        if len(pdf.pages) > 50:
            raise ValueError("Поддерживаются PDF до 50 страниц.")
        for page in pdf.pages:
            top = 0
            for table in sorted(page.find_tables(), key=lambda item: item.bbox[1]):
                if table.bbox[1] > top:
                    before_table = page.crop((0, top, page.width, table.bbox[1]))
                    yield from (before_table.extract_text() or "").splitlines()
                yield table.extract()
                top = max(top, table.bbox[3])
            if top < page.height:
                after_table = page.crop((0, top, page.width, page.height))
                yield from (after_table.extract_text() or "").splitlines()


def _columns(row):
    columns = {}
    for i, cell in enumerate(row):
        label = " ".join(cell.lower().split())
        if label in {"время", "тайминг", "таймкод"}:
            columns["time"] = i
        elif label in {"что происходит", "изображение", "видеоряд", "описание сцены"}:
            columns["visual"] = i
        elif label in {"голос", "реплики", "озвучка", "речь", "звук", "голос / звук"}:
            columns["speech"] = i
        elif label in {"экранный текст", "текст на экране", "супер"}:
            columns["screen_text"] = i
    return columns if {"time", "visual"} <= columns.keys() else None


def _document_card(blocks, name, data, *, pdf):
    card = {
        "id": "doc-" + hashlib.sha256(data).hexdigest(),
        "name": Path(name).stem,
        "scenes": [],
    }
    context = {}
    section = None
    columns = None
    duration = None
    final_screen = ""
    for block in blocks:
        if isinstance(block, str):
            line = block.strip()
            timing = DURATION.fullmatch(line)
            if timing:
                duration = _seconds(timing[1])
                continue
            screen = FINAL_SCREEN.fullmatch(line)
            if screen:
                final_screen = screen[1]
                continue
            heading = line.lower().replace("ё", "е").rstrip(":")
            if heading in CONTEXT_HEADINGS:
                section = CONTEXT_HEADINGS[heading]
                context.setdefault(section, [])
                columns = None
            elif heading in {"сценарий", "ключевая идея"} or re.match(
                r"^(?:сообщение|инсайт|категории|тон|хук)\s*:", heading
            ):
                section = None
            elif section and line:
                context[section].append(line)
            continue

        rows = [[(cell or "").strip() for cell in row] for row in block]
        if pdf and columns and not section and not any(
            _columns(row) or (
                len(row) > columns["time"] and INTERVAL.fullmatch(row[columns["time"]])
            )
            for row in rows
        ):
            continue
        for row in rows:
            header = _columns(row)
            if header:
                columns = header
                section = None
                continue
            if section:
                context[section].append(" | ".join(filter(None, row)))
                continue
            if not columns or max(columns.values()) >= len(row) or not any(row):
                continue
            interval = INTERVAL.fullmatch(row[columns["time"]])
            scene = {
                key: row[columns[key]] if key in columns else ""
                for key in ("visual", "speech", "screen_text")
            }
            if not interval:
                if pdf and not row[columns["time"]] and card["scenes"]:
                    previous = card["scenes"][-1]
                    for key, value in scene.items():
                        if value:
                            previous[key] = "\n".join(filter(None, [previous[key], value]))
                    continue
                raise ValueError(f"Не удалось прочитать тайминг «{row[columns['time']]}».")
            card["scenes"].append(
                {"start": _seconds(interval[1]), "end": _seconds(interval[2]), **scene}
            )

    if not card["scenes"]:
        message = "Не найдена таблица сцен с колонками «Время» и «Что происходит»."
        if pdf:
            message += " Для скана или PDF без читаемой таблицы загрузите исходный DOCX."
        raise ValueError(message)
    for scene in card["scenes"]:
        label = SCREEN_LABEL.search(scene["visual"])
        if label:
            visual = scene["visual"]
            scene["visual"] = visual[:label.start()].rstrip()
            scene["screen_text"] = "\n".join(
                filter(None, [scene["screen_text"], visual[label.end():].strip()])
            )
    last_scene = card["scenes"][-1]
    if final_screen and final_screen not in last_scene["screen_text"]:
        last_scene["screen_text"] = "\n".join(
            filter(None, [last_scene["screen_text"], final_screen])
        )
    card["duration_sec"] = card["scenes"][-1]["end"] if duration is None else duration
    card.update({key: "\n".join(parts) for key, parts in context.items() if parts})
    return card


def parse_upload(name, data):
    suffix = Path(name).suffix.lower()
    try:
        if len(data) > 5_000_000:
            raise ValueError("Размер файла превышает 5 МБ.")
        if suffix == ".json":
            return parse_cards(data.decode("utf-8-sig"))
        if suffix == ".docx":
            blocks = _word_blocks(data)
        elif suffix == ".pdf":
            blocks = _pdf_blocks(data)
        else:
            raise ValueError("Выберите файл DOCX, PDF или JSON.")
        return load_cards(_document_card(blocks, name, data, pdf=suffix == ".pdf"))
    except ValueError as exc:
        raise ValueError(f"{name}: {exc}") from exc
    except (OSError, KeyError, BadZipFile, ElementTree.ParseError) as exc:
        raise ValueError(f"{name}: не удалось прочитать документ.") from exc
