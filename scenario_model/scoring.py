# coding: utf-8
import math
import numpy as np
from .config import FEATURES, ROOT, read_json, verify_bundle


class ScenarioRidge:
    def __init__(self):
        self.bundle_hash = verify_bundle()
        self.state = read_json(ROOT / "artifacts/ridge.json")
        if self.state["columns"] != FEATURES:
            raise ValueError("Неверный порядок признаков.")

    def predict(self, rows):
        s = self.state
        x = np.array(
            [[self._number(row[c]) for c in FEATURES] for row in rows], dtype=float
        )
        if s["panel_consensus"]:
            for j, c in enumerate(FEATURES):
                if c.startswith(("panel__", "fresh__")):
                    x[:, j] = np.floor(x[:, j] + 0.5)
        x = np.where(np.isfinite(x), x, np.asarray(s["median"]))
        z = (x - np.asarray(s["center"])) / np.asarray(s["scale"])
        step = s["step"]
        if step:
            z = np.round(z / step) * step
        p = np.column_stack([np.ones(len(z)), z]) @ np.asarray(s["estimator"])
        p = np.exp(np.clip(p * s["target_scale"] + s["target_mean"], -12, 3))
        return np.clip(
            s["calibration_intercept"] + s["calibration_slope"] * p, 0.001, 1.0
        )

    @staticmethod
    def _number(value):
        if str(value).lower() == "true":
            return 1.0
        if str(value).lower() == "false":
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return float("nan")

    def summarize(self, individual_rows):
        if len(individual_rows) != 5:
            raise ValueError("Для расчёта нужны все пять оценщиков.")
        # Raw competition remains available in individual_rows; only this input is pooled.
        competition = float(
            np.mean([r["message_competition_mean"] for r in individual_rows])
        )
        rows = [{**r, "message_competition_mean": competition} for r in individual_rows]
        p = self.predict(rows) * 100
        score = float(p.mean())
        return {
            "score": score,
            "class": (
                "Низкий" if score <= 20 else "Средний" if score <= 26 else "Высокий"
            ),
            "assessor_scores": p.tolist(),
            "score_range": [float(p.min()), float(p.max())],
            "model_inputs": rows,
            "feature_means": {
                c: float(np.mean([r[c] for r in rows])) for c in FEATURES
            },
            "feature_ranges": {
                c: [
                    float(min(r[c] for r in individual_rows)),
                    float(max(r[c] for r in individual_rows)),
                ]
                for c in FEATURES
            },
        }
