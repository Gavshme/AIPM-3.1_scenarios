"""Display the existing feature contributions without changing model results."""

import math
import textwrap

from .config import LABELS


def contribution_chart(feature_means, contributions):
    rows = []
    for feature, score in feature_means.items():
        delta = float(contributions[feature])
        if not math.isfinite(delta):
            raise ValueError("Вклад признака должен быть конечным числом.")
        rounded = round(delta, 2)
        rows.append({
            "feature": LABELS[feature],
            "label": "\n".join(textwrap.wrap(LABELS[feature], 29, break_long_words=False)),
            "score": float(score),
            "contribution": delta,
            "display": (f"{rounded:+.2f}".replace("-", "−") if rounded else "0") + " п.п.",
            "direction": "Повышает MD" if rounded > 0 else "Снижает MD" if rounded < 0 else "Нейтрально",
        })
    rows.sort(key=lambda row: row["contribution"], reverse=True)
    extent = max(1.0, max((abs(row["contribution"]) for row in rows), default=0) * 1.35)
    y = {"field": "label", "type": "nominal", "sort": [row["label"] for row in rows],
         "axis": {"title": None, "domain": False, "ticks": False, "labelPadding": 12,
                  "labelLimit": 240, "labelFontSize": 12, "labelLineHeight": 15,
                  "labelExpr": "split(datum.label, '\\n')"}}
    x = {"field": "contribution", "type": "quantitative",
         "scale": {"domain": [-extent, extent], "nice": False},
         "axis": {"title": "Вклад в MD, п.п.", "titlePadding": 14, "domain": False,
                  "gridColor": "#E7ECF2", "gridDash": [2, 4], "tickCount": 5,
                  "format": "+.1f", "labelColor": "#66758A", "titleColor": "#66758A"}}
    tooltip = [{"field": "feature", "title": "Признак", "type": "nominal"},
               {"field": "score", "title": "Средняя оценка", "type": "quantitative", "format": ".2f"},
               {"field": "contribution", "title": "Вклад в MD, п.п.", "type": "quantitative", "format": "+.2f"}]
    layers = [
        {"data": {"values": [{}]}, "mark": {"type": "rule", "color": "#8D9BAE", "strokeWidth": 1.2},
         "encoding": {"x": {"datum": 0, "type": "quantitative"}}},
        {"mark": {"type": "bar", "cornerRadius": 5, "size": 21},
         "encoding": {"x": x, "y": y, "tooltip": tooltip,
                      "color": {"field": "direction", "type": "nominal",
                                "scale": {"domain": ["Повышает MD", "Снижает MD", "Нейтрально"],
                                          "range": ["#278568", "#CE6663", "#8793A5"]},
                                "legend": {"title": None, "orient": "top", "direction": "horizontal",
                                           "labelFontSize": 12, "symbolType": "circle", "padding": 12}}}},
        {"transform": [{"filter": "datum.direction === 'Нейтрально'"}],
         "mark": {"type": "point", "filled": True, "size": 45, "color": "#8793A5"},
         "encoding": {"x": x, "y": y, "tooltip": tooltip}},
    ]
    for condition, align, offset in [("datum.contribution >= 0", "left", 8), ("datum.contribution < 0", "right", -8)]:
        layers.append({
            "transform": [{"filter": condition}],
            "mark": {"type": "text", "align": align, "dx": offset, "fontSize": 12,
                     "fontWeight": 600, "color": "#31445D"},
            "encoding": {"x": x, "y": y, "text": {"field": "display"}, "tooltip": tooltip},
        })
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": rows}, "height": max(240, len(rows) * 52),
        "padding": {"top": 8, "bottom": 8, "left": 8, "right": 12},
        "layer": layers,
        "config": {"view": {"stroke": None}, "font": "Arial", "background": "transparent",
                   "axis": {"labelFont": "Arial", "titleFont": "Arial"}},
    }
