# coding: utf-8
"""Export trusted local research artifacts; never copy cards, targets or training histories."""
from pathlib import Path
import argparse, hashlib, json, importlib.util, sys
import joblib, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scenario_model.config import FEATURES, MODELS


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def seal():
    files = [
        ROOT / "artifacts/ridge.json",
        ROOT / "artifacts/metrics.json",
        ROOT / "artifacts/provenance.json",
        ROOT / "protocols/stages.json",
    ]
    write(
        ROOT / "artifacts/manifest.json",
        {"files_sha256": {str(p.relative_to(ROOT)): sha(p) for p in files}},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, required=True)
    args = ap.parse_args()
    W = args.workspace.resolve()
    B = W / "outputs/scenario_selected_five_benefit_groups_20261102"
    sel = json.loads((B / "selection.json").read_text())
    for f, h in json.loads((B / "manifest.json").read_text())["files_sha256"].items():
        assert sha(B / f) == h, f
    assert sel["feature_columns"] == FEATURES and sel["assessors"] == MODELS
    path = Path(sel["ridge_path"])
    state = json.loads((path / "state.json").read_text())
    sp = state["spec"]
    assert (
        sp["task"] == "m"
        and sp["learner"] == "ridge"
        and sp["alpha"] == 1
        and sp["target"] == "log"
        and sp["mode"] == "expanded"
        and sp["noticeability_weight_power"] == 0
    )
    clean = {
        k: state[k]
        for k in [
            "columns",
            "median",
            "center",
            "scale",
            "target_mean",
            "target_scale",
            "calibration_intercept",
            "calibration_slope",
        ]
    }
    clean.update(
        estimator=joblib.load(path / "estimator.joblib").tolist(),
        step=sp["step"],
        panel_consensus=sp["panel_consensus"],
        model_id=sel["version"],
        target="log",
        alpha=1,
    )
    write(ROOT / "artifacts/ridge.json", clean)

    def stage(body, marker="\nSCENARIOS:\n"):
        return {
            "model": body["model"],
            "parameters": {
                k: v
                for k, v in body.items()
                if k not in ["model", "messages", "response_format"]
            },
            "response_format": body["response_format"],
            "prompt": body["messages"][0]["content"].split(marker)[0],
            "marker": marker,
        }

    c = json.loads((W / "outputs/scenario_aipm3_20261013/contract.json").read_text())
    claim = {
        "model": c["model"],
        "parameters": {"temperature": c["temperature"]},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "scenario_md9",
                "strict": True,
                "schema": c["schema"],
            },
        },
        "prompt": c["prompt"],
        "marker": "\nСценарии:\n",
        "repeats": 3,
    }
    specs = []
    for local, source in enumerate([0, 1, 2, 8, 9]):
        stages = {}
        for name, folder in [("common", "seven_features"), ("precision", "precision")]:
            parent = (
                W
                / "outputs/scenario_creative_context_audio_20261102/requests"
                / folder
                / "short"
                if source < 3
                else W
                / "outputs/scenario_assessor_count_3to10_20261102/requests"
                / folder
            )
            j = json.loads((parent / f"model_{source}_batch_00.json").read_text())
            stages[name] = stage(j["body"])
        j = json.loads(
            (
                W
                / "outputs/scenario_benefit_groups_five_v3_20261102/requests/pilot"
                / f"model_{local}_batch_00.json"
            ).read_text()
        )
        stages["benefits"] = stage(j["body"])
        assert all(s["model"] == MODELS[local] for s in stages.values())
        specs.append({"model": MODELS[local], "stages": stages})
    write(
        ROOT / "protocols/stages.json",
        {
            "protocol_id": "scenario-blind-claim-five-v1",
            "claim": claim,
            "assessors": specs,
            "aggregation": "Exact-string modal claim across3; earliest tie. Per-assessor benefits grouped, min4. Raw competition averaged across5; other6individual. Mean5Ridgepredictions.",
        },
    )
    write(
        ROOT / "artifacts/metrics.json",
        {
            "n_scenarios": 58,
            "n_families": 56,
            "metrics_oof": sel["metrics_oof"],
            "independent_validation": False,
            "note": "Historical research metrics, not a new live end-to-end validation of this app.",
        },
    )
    provenance = {
        "source_release": sel["version"],
        "source_manifest_sha256": sha(B / "manifest.json"),
        "source_estimator_sha256": sha(path / "estimator.joblib"),
        "source_state_sha256": sha(path / "state.json"),
        "training_data_included": False,
        "training_performed": False,
        "claim_stage": "Historical combined9feature prompt retained to preserve how initial claims were read; only main_claim is used.3Gemini reads, exact-string mode, earliest tie; scenes only. Singleton transport and current cards require prospective evaluation.",
    }
    write(ROOT / "artifacts/provenance.json", provenance)
    seal()
    spec = importlib.util.spec_from_file_location("original_model", B / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old = module.Head.load(path)
    from scenario_model.scoring import ScenarioRidge

    exported = ScenarioRidge()
    f = pd.read_csv(B / "features_repeats_290.csv")
    oldp = old.predict(f)
    newp = exported.predict(f.to_dict("records"))
    np.testing.assert_allclose(oldp, newp, rtol=0, atol=1e-12)
    # Probe half-rounding, missing values and clipping outside the observed rows.
    rng = np.random.default_rng(71)
    probe = pd.DataFrame(rng.uniform(-4, 8, size=(256, 7)), columns=FEATURES)
    probe.iloc[0, 2] = np.nan
    probe.iloc[1, 3] = 0.5
    probe.iloc[2, 3] = 1.5
    np.testing.assert_allclose(
        old.predict(probe),
        exported.predict(probe.to_dict("records")),
        rtol=0,
        atol=1e-12,
    )
    provenance["parity"] = {
        "research_rows": 290,
        "scenarios": 58,
        "additional_probes": 256,
        "max_abs_error_fraction": float(np.max(abs(oldp - newp))),
        "absolute_tolerance_fraction": 1e-12,
    }
    write(ROOT / "artifacts/provenance.json", provenance)
    seal()
    print(json.dumps(provenance["parity"], indent=2))


if __name__ == "__main__":
    main()
