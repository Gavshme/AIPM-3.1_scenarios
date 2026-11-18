"""Export the user-selected 2026-11-18 research artifacts without training at runtime.

Developer-only dependencies: numpy, pandas, scipy, scikit-learn and joblib.
Usage: python tools/export_nusvr_ordinal.py --research-root /path/to/research
Raw scenarios, responses, target rows and family identities are not exported.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--research-root", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    research = args.research_root.resolve() / "outputs"
    score_source = research / "scenario_nusvr_role_reveal_20261118"
    class_source = research / "scenario_joint_calibration_methods_20261118"
    state = read(score_source / "models/nusvr9_role/state.json")
    model = joblib.load(score_source / "models/nusvr9_role/estimator.joblib")
    score = {
        "model_id": "scenario-nusvr9-four-20261118",
        "columns": state["features"],
        **{key: state[key] for key in ("median", "center", "scale", "target_mean", "target_scale")},
        "step": state["parameters"]["step"],
        "kernel": "rbf", "gamma": float(model._gamma),
        "support_vectors": model.support_vectors_.tolist(),
        "dual_coef": model.dual_coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "calibration_intercept": state["intercept"],
        "calibration_slope": state["slope"],
        "training_target": "human MD", "training_scenarios": 58, "training_families": 56,
        "parameters": state["parameters"],
    }
    save(root / "artifacts/nusvr9.json", score)
    selected = read(class_source / "direct_ordinal8_model.json")
    assessors = read(research / "scenario_active_selection.json")["models"]["short"]["app_protocol"]["assessors"]
    classifier = {
        **{key: selected[key] for key in ("features", "penalty", "scaler_mean", "scaler_scale", "weights", "thresholds", "human_cutoffs")},
        "model_id": "scenario-joint-ordinal8-four-20261118",
        "assessors": assessors, "default_class_rule": "posterior_mode",
        "probability_calibration": "none", "training_target": "video and human MD classes",
        "target_weights": {"video": .5, "human": .5},
        "video_cutoffs": [19.55303191060736, 28.61248470469908],
        "training_scenarios": 56, "training_families": 54,
    }
    save(root / "artifacts/joint_ordinal8.json", classifier)
    extraction = read(research / "scenarios_video_model_20261118/contract.json")
    save(root / "artifacts/role_inputs.json", {
        "protocol_id": "scenario-video-inputs-three-four-20261118-v1",
        "marker": "\nSCENARIOS:\n", "prompt": extraction["prompt"],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "scenario_video_md_missing3", "strict": True, "schema": extraction["schema"],
        }},
    })
    score_metrics = read(score_source / "metrics.json")
    class_metrics = read(class_source / "direct_ordinal_metrics.json")
    ordinal_metrics = next(row["details"] for row in class_metrics["results"] if row["model"] == "direct_ordinal8" and row["rule"] == "mode")
    save(root / "artifacts/release_metrics.json", {
        "version": "scenario-nusvr9-joint-ordinal8-four-20261118-v1",
        "n_common_pairs": 56, "n_common_families": 54,
        "score_oof": score_metrics["models"]["nusvr9_role"]["oof_on_common_56"],
        "score_validation": score_metrics["validation"],
        "class_oof": ordinal_metrics,
        "class_validation": class_metrics["validation"],
        "independent_new_holdout": False,
        "video_reference": "aipm-0719-refined-20261117; published full-fit video scores/classes",
        "limitation": "Research cohort repeatedly used in model experiments; OOF is not independent confirmation.",
    })
    # Synthetic numerical parity fixtures contain no real scenario or human-rating rows.
    rng = np.random.default_rng(42)
    upper = [1, 1, 4, 3, 1, 4, 3, 2, 4]
    fixtures = []
    for i in range(32):
        raw = rng.integers(0, np.asarray(upper) + 1, size=(4, 9)).astype(float)
        if i < 16:
            raw[:] = raw[0]  # Include extremes as well as four-assessor averages.
        if i == 31:
            # Deliberately exercise the selected mode rule where a median would disagree.
            for _ in range(10000):
                raw = rng.integers(0, np.asarray(upper) + 1, size=(4, 9)).astype(float)
                mean = raw[:, :8].mean(axis=0)
                eta = ((mean - selected["scaler_mean"]) / selected["scaler_scale"]) @ np.asarray(selected["weights"])
                cumulative = expit(np.asarray(selected["thresholds"]) - eta)
                probabilities = np.diff(np.r_[0., cumulative, 1.])
                if int(probabilities.argmax()) != int((np.cumsum(probabilities) < .5).sum()):
                    break
            else:
                raise ValueError("Could not construct the mode-versus-median fixture")
        x = raw.copy()
        x[:, 6] = raw[:, 6].mean()
        z = (x - state["center"]) / state["scale"]
        z = np.round(z / state["parameters"]["step"]) * state["parameters"]["step"]
        predictions = np.exp(np.clip(model.predict(z) * state["target_scale"] + state["target_mean"], -12, 3))
        predictions = np.clip(state["intercept"] + state["slope"] * predictions, .001, 1.) * 100
        mean = raw[:, :8].mean(axis=0)
        eta = ((mean - selected["scaler_mean"]) / selected["scaler_scale"]) @ np.asarray(selected["weights"])
        cumulative = expit(np.asarray(selected["thresholds"]) - eta)
        p = np.diff(np.r_[0., cumulative, 1.])
        fixtures.append({"id": f"synthetic_{i:02}", "rows": [dict(zip(state["features"], row)) for row in raw.tolist()],
                         "assessor_scores": predictions.tolist(), "mean_score": float(predictions.mean()),
                         "class_index": int(p.argmax()), "class_probabilities": p.tolist()})
    save(root / "tests/fixtures/nusvr_ordinal.json", fixtures)
    source_files = [score_source / "models/nusvr9_role/state.json", score_source / "models/nusvr9_role/estimator.joblib",
                    class_source / "direct_ordinal8_model.json", research / "scenarios_video_model_20261118/contract.json"]
    save(root / "artifacts/release_provenance.json", {
        "version": "scenario-nusvr9-joint-ordinal8-four-20261118-v1", "selected_by_user": "2026-11-18",
        "source_sha256": {str(path.relative_to(research)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
        "export": "numeric frozen parameters and synthetic parity fixtures only; no inference fitting",
    })
    paths = list(read(root / "artifacts/manifest.json")["files_sha256"])
    paths += ["artifacts/author_message.json", "artifacts/function_demo.json", "artifacts/nusvr9.json",
              "artifacts/joint_ordinal8.json", "artifacts/role_inputs.json", "artifacts/release_metrics.json",
              "artifacts/release_provenance.json", "tests/fixtures/nusvr_ordinal.json"]
    save(root / "artifacts/nusvr_manifest.json", {
        "version": "scenario-nusvr9-joint-ordinal8-four-20261118-v1",
        "files_sha256": {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths},
    })
    print("Exported NuSVR, joint ordinal class model, extraction contract and 32 synthetic fixtures.")


if __name__ == "__main__":
    main()
