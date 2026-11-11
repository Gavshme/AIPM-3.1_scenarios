# coding: utf-8
import numpy as np

from .config import ACTIVE_MODELS, DEMO_FEATURES, ROOT, digest, read_json


CLASSES = ("Низкий", "Средний", "Высокий")


def classify_features(feature_means):
    model = read_json(ROOT / "artifacts/ordinal_classes.json")
    columns = model["features"]
    if (
        columns != DEMO_FEATURES
        or model["assessors"] != ACTIVE_MODELS
        or model["default_class_rule"] != "posterior_median"
    ):
        raise ValueError("Классификатор несовместим с признаками или составом оценщиков.")
    features = np.array([feature_means[col] for col in columns], dtype=float)
    means = np.asarray(model["scaler_mean"], dtype=float)
    scale = np.asarray(model["scaler_scale"], dtype=float)
    weights = np.asarray(model["weights"], dtype=float)
    thresholds = np.asarray(model["thresholds"], dtype=float)
    if (
        any(a.shape != features.shape for a in (means, scale, weights))
        or not np.all(scale > 0)
        or thresholds.shape != (2,)
        or not thresholds[0] < thresholds[1]
        or not all(np.isfinite(a).all() for a in (features, means, scale, weights, thresholds))
    ):
        raise ValueError("Некорректные параметры порядкового классификатора.")
    location = np.dot((features - means) / scale, weights)
    logits = thresholds - location
    cumulative = 1 / (1 + np.exp(-np.clip(logits, -700, 700)))
    index = int(np.count_nonzero(cumulative < 0.5))
    return {
        "class": CLASSES[index],
        "model_id": model["model_id"],
        "model_hash": digest(model),
        "probabilities": dict(zip(CLASSES, np.diff(np.r_[0.0, cumulative, 1.0]).tolist())),
    }
