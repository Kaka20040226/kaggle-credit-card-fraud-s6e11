import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(config_path):
    config_path = Path(config_path).resolve()
    root = config_path.parent.parent
    config = json.loads(config_path.read_text())
    paths = {name: root / value for name, value in config["paths"].items()}

    for name, expected in config["expected_sha256"].items():
        actual = sha256(paths[name])
        if actual != expected:
            raise ValueError(f"SHA256 mismatch for {name}: {actual}")

    target = config["target"]
    train = pd.read_csv(paths["train"], usecols=[target])
    test = pd.read_csv(paths["test"], usecols=["id"])
    sample = pd.read_csv(paths["sample_submission"])
    if list(sample.columns) != ["id", target]:
        raise ValueError(f"Unexpected sample columns: {list(sample.columns)}")
    if not test["id"].equals(sample["id"]):
        raise ValueError("Test IDs do not match the sample submission")

    y = train[target].to_numpy()
    wide_oof = np.load(paths["wide_lm_oof"], allow_pickle=False).astype(float)
    wide_test = np.load(paths["wide_lm_test"], allow_pickle=False).astype(float)
    part9_test = np.load(paths["part9_test"], allow_pickle=False).astype(float)
    lm_oof = np.load(paths["lm_oof"], allow_pickle=False).astype(float)
    if wide_oof.shape != y.shape or lm_oof.shape != y.shape:
        raise ValueError("OOF array length does not match train")
    if wide_test.shape != (len(test),) or part9_test.shape != (len(test),):
        raise ValueError("Test prediction length does not match test")
    if not all(np.isfinite(a).all() for a in (wide_oof, lm_oof, wide_test, part9_test)):
        raise ValueError("Predictions contain NaN or infinity")

    weights = config["weights"]
    n_test = len(test)
    prediction = rankdata(
        weights["wide_lm_stack_test"] * rankdata(wide_test) / n_test
        + weights["part9_test"] * rankdata(part9_test) / n_test
    ) / n_test
    members = pd.read_csv(paths["members"])
    splitter = StratifiedKFold(
        n_splits=config["validation"]["n_splits"],
        shuffle=True,
        random_state=config["validation"]["seed"]
    )
    fold_counts = {fold: len(valid) for fold, (_, valid) in enumerate(splitter.split(np.zeros(len(y)), y))}
    artifact_dir = paths["artifact_dir"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths["submission"].parent.mkdir(parents=True, exist_ok=True)
    paths["metrics_log"].parent.mkdir(parents=True, exist_ok=True)

    np.save(artifact_dir / "oof_wide_lm_stack.npy", wide_oof.astype(np.float32))
    np.save(artifact_dir / "test_predictions.npy", prediction.astype(np.float64))
    submission = pd.DataFrame({"id": test["id"], target: prediction})
    submission.to_csv(paths["submission"], index=False)
    submission_hash = sha256(paths["submission"])
    if submission_hash != config["expected_submission_sha256"]:
        raise ValueError(f"Submission SHA256 mismatch: {submission_hash}")

    metrics = {
        "run_id": config["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "wide_lm_stack_oof_auc": float(roc_auc_score(y, wide_oof)),
        "lm_member_oof_auc": float(roc_auc_score(y, lm_oof)),
        "source_reported_final_blend_nested_auc": config["validation"]["source_reported_final_blend_nested_auc"],
        "source_reported_final_blend_fold_auc": config["validation"]["source_reported_final_blend_fold_auc"],
        "wide_part9_test_spearman": float(np.corrcoef(rankdata(wide_test), rankdata(part9_test))[0, 1]),
        "member_count": int(len(members)),
        "fold_counts": {str(k): v for k, v in fold_counts.items()},
        "source_rebuild_exact_hash": True,
        "submission_rows": int(len(submission)),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max())
    }
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    paths["metrics_log"].write_text(json.dumps(metrics, sort_keys=True) + "\n")

    tracked = {
        "config": config_path,
        "train": paths["train"],
        "test": paths["test"],
        "sample_submission": paths["sample_submission"],
        "wide_lm_oof": paths["wide_lm_oof"],
        "wide_lm_test": paths["wide_lm_test"],
        "part9_test": paths["part9_test"],
        "lm_oof": paths["lm_oof"],
        "members": paths["members"],
        "submission": paths["submission"],
        "oof_artifact": artifact_dir / "oof_wide_lm_stack.npy",
        "test_artifact": artifact_dir / "test_predictions.npy"
    }
    manifest = {
        "run_id": config["run_id"],
        "competition": config["competition"],
        "source": config["source"],
        "weights": weights,
        "files": {name: {"path": str(path.relative_to(root)), "sha256": sha256(path)} for name, path in tracked.items()},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__
        }
    }
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"submission": str(paths["submission"]), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python src/build_v2_part12.py configs/v2_part12.json")
    main(sys.argv[1])
