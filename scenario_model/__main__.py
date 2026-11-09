# coding: utf-8
import argparse, json, os, sys
from pathlib import Path
from .cards import parse_cards
from .client import LlmClient, atomic_json
from .pipeline import evaluate_card, replay_result


def main():
    ap = argparse.ArgumentParser(
        description="Оценка сценариев: AI-посыл → пять оценщиков → Ridge MD."
    )
    sub = ap.add_subparsers(dest="command", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("input", type=Path)
    e.add_argument("--output", type=Path, required=True)
    e.add_argument(
        "--live",
        action="store_true",
        help="Разрешить новые API-запросы; по умолчанию только сохранённые ответы.",
    )
    e.add_argument(
        "--cache", default=os.environ.get("SCENARIO_CACHE_DIR", ".scenario_cache")
    )
    r = sub.add_parser("replay")
    r.add_argument("result", type=Path)
    args = ap.parse_args()
    try:
        if args.command == "replay":
            data = json.loads(args.result.read_text(encoding="utf-8"))
            items = data.get("results", [data])
            if not items:
                raise ValueError("В файле нет завершённых оценок.")
            for result in items:
                score = replay_result(result)
                print(result["card"]["id"], f"{score['score']:.3f}", score["class"])
            print("Проверено без API и переобучения.")
            return 0
        cards = parse_cards(args.input.read_text(encoding="utf-8-sig"))
        client = LlmClient(cache_dir=args.cache, allow_live=args.live)
        out = {"format": "scenario-batch-v1", "results": [], "errors": []}
        for item in cards:
            try:
                result = evaluate_card(
                    item["original"],
                    client,
                    progress=lambda text: print(
                        item["card"]["id"] + ": " + text, flush=True
                    ),
                )
                out["results"].append(result)
                print(
                    f"{result['scoring']['score']:.3f} · {result['scoring']['class']}"
                )
            except Exception as exc:
                out["errors"].append({"id": item["card"]["id"], "error": str(exc)})
                print(str(exc), file=sys.stderr)
            atomic_json(args.output, out)
        return int(bool(out["errors"]))
    except (ValueError, KeyError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
