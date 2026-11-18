# coding: utf-8
import math
import numpy as np
from .config import DEMO_FEATURES, FEATURES, NUSVR_FEATURES, ROOT, digest, read_json, verify_bundle, verify_nusvr_bundle


class ScenarioRidge:
    def __init__(self, author_bundle=None):
        self.bundle_hash = verify_bundle()
        if author_bundle is None:
            self.state = read_json(ROOT / "artifacts/ridge.json")
        else:
            self.state = author_bundle["ridge"]
            self.bundle_hash = digest([self.bundle_hash, author_bundle])
        self.columns = self.state["columns"]
        expected = DEMO_FEATURES if author_bundle and "function_demo" in author_bundle else FEATURES
        if self.columns != expected:
            raise ValueError("Неверный порядок признаков.")

    def predict(self, rows):
        s = self.state
        x = np.array(
            [[self._number(row[c]) for c in self.columns] for row in rows], dtype=float
        )
        if s["panel_consensus"]:
            for j, c in enumerate(self.columns):
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

    def summarize(self, individual_rows, *, assessor_count=5):
        if len(individual_rows) != assessor_count:
            raise ValueError(f"Для расчёта нужны оценки всех моделей ({assessor_count}).")
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
                c: float(np.mean([r[c] for r in rows])) for c in self.columns
            },
            "feature_ranges": {
                c: [
                    float(min(r[c] for r in individual_rows)),
                    float(max(r[c] for r in individual_rows)),
                ]
                for c in self.columns
            },
        }

    def feature_contributions(self, rows):
        """Explain the mean MD in score points relative to the training medians."""
        reference = dict(zip(self.columns, self.state["median"]))
        n = len(self.columns)
        hybrids = [
            {
                column: row[column] if mask & (1 << i) else reference[column]
                for i, column in enumerate(self.columns)
            }
            for mask in range(1 << n)
            for row in rows
        ]
        values = self.predict(hybrids).reshape(-1, len(rows)).mean(axis=1) * 100
        weights = [1 / (n * math.comb(n - 1, k)) for k in range(n)]
        contributions = {}
        for i, column in enumerate(self.columns):
            bit = 1 << i
            contributions[column] = sum(
                weights[bin(mask).count("1")] * (values[mask | bit] - values[mask])
                for mask in range(1 << n)
                if not mask & bit
            )
        return float(values[0]), contributions


class ScenarioNuSVR(ScenarioRidge):
    """Frozen RBF NuSVR, exported as numeric JSON; no fitting at runtime."""

    def __init__(self):
        self.bundle_hash = verify_nusvr_bundle()
        self.state = read_json(ROOT / "artifacts/nusvr9.json")
        self.columns = self.state["columns"]
        s = self.state
        if self.columns != NUSVR_FEATURES or s["kernel"] != "rbf":
            raise ValueError("Несовместимая модель NuSVR.")
        n = len(self.columns)
        support = np.asarray(s["support_vectors"], dtype=float)
        dual = np.asarray(s["dual_coef"], dtype=float)
        vectors = [np.asarray(s[key], dtype=float) for key in ("median", "center", "scale")]
        scalars = [s[key] for key in ("gamma", "intercept", "target_scale", "target_mean", "calibration_intercept", "calibration_slope", "step")]
        if (
            support.ndim != 2 or support.shape[1] != n or support.shape[0] == 0
            or dual.shape != (support.shape[0],)
            or any(a.shape != (n,) or not np.isfinite(a).all() for a in vectors)
            or not np.isfinite(support).all() or not np.isfinite(dual).all()
            or not np.isfinite(scalars).all() or not np.all(vectors[2] > 0)
            or s["gamma"] <= 0 or s["target_scale"] <= 0 or s["step"] < 0
            or s["calibration_slope"] <= 0
        ):
            raise ValueError("Некорректные параметры NuSVR.")

    def predict(self, rows):
        s = self.state
        x = np.asarray([[self._number(row[c]) for c in self.columns] for row in rows], dtype=float)
        x = np.where(np.isfinite(x), x, np.asarray(s["median"]))
        z = (x - np.asarray(s["center"])) / np.asarray(s["scale"])
        if s["step"]:
            z = np.round(z / s["step"]) * s["step"]
        support = np.asarray(s["support_vectors"])
        squared_distance = np.maximum(
            (z * z).sum(axis=1)[:, None] + (support * support).sum(axis=1)[None, :] - 2 * z @ support.T,
            0.0,
        )
        p = np.exp(-s["gamma"] * squared_distance) @ np.asarray(s["dual_coef"]) + s["intercept"]
        p = np.exp(np.clip(p * s["target_scale"] + s["target_mean"], -12, 3))
        return np.clip(s["calibration_intercept"] + s["calibration_slope"] * p, .001, 1.)
