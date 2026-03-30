from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

import pandas as pd

from scr.Model.predict_models import evaluate_model, predict_admission
from scr.Model.train_models import load_model, save_model, train_mlp_model
from scr.data.make_dataset import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_admission_data,
    load_prepare_split,
)

warnings.filterwarnings(
    "ignore",
    message=".*encountered in matmul.*",
    category=RuntimeWarning,
)


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str


def run_check(name: str, fn: Callable[[], str]) -> CheckResult:
    try:
        details = fn()
        return CheckResult(name=name, passed=True, details=details)
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name=name, passed=False, details=f"{type(exc).__name__}: {exc}"
        )


def resolve_data_path(project_root: Path) -> Path:
    """Locate the dataset in the current project layout, with legacy fallback."""
    preferred = project_root / "data" / "raw" / "Admission.csv"
    if preferred.exists():
        return preferred

    legacy = project_root / "Admission.csv"
    if legacy.exists():
        return legacy

    raise FileNotFoundError(
        "Dataset not found. Expected data/raw/Admission.csv (or legacy Admission.csv)."
    )


def verify_required_files(project_root: Path) -> str:
    dataset_candidates = [
        project_root / "data" / "raw" / "Admission.csv",
        project_root / "Admission.csv",
    ]
    streamlit_candidates = [
        project_root / "streamlit_app.py",
        project_root / "streamlit_04NeuralNetworks_app.py",
    ]
    required = [
        project_root / "main.py",
        project_root / "scr" / "data" / "make_dataset.py",
        project_root / "scr" / "Model" / "train_models.py",
        project_root / "scr" / "Model" / "predict_models.py",
    ]

    missing = [str(p.relative_to(project_root)) for p in required if not p.exists()]
    if not any(p.exists() for p in dataset_candidates):
        missing.append("data/raw/Admission.csv (or Admission.csv)")
    if not any(p.exists() for p in streamlit_candidates):
        missing.append("streamlit_app.py (or streamlit_04NeuralNetworks_app.py)")
    if missing:
        raise FileNotFoundError(f"Missing files: {', '.join(missing)}")

    return f"{len(required) + 2} required files are present"


def verify_dataset_schema(project_root: Path) -> str:
    csv_path = resolve_data_path(project_root)
    df = load_admission_data(csv_path)

    expected = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = sorted(expected - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    if df.empty:
        raise ValueError("Admission.csv is empty")

    null_counts = df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().sum().sum()
    return f"rows={len(df)}, cols={len(df.columns)}, missing_core_values={int(null_counts)}"


def verify_data_prep_and_split(project_root: Path) -> str:
    X_train, X_test, y_train, y_test, _ = load_prepare_split(
        resolve_data_path(project_root)
    )

    if list(X_train.columns) != FEATURE_COLUMNS:
        raise ValueError("Feature columns mismatch after preprocessing")

    train_ratio = len(X_train) / (len(X_train) + len(X_test))
    if not (0.75 <= train_ratio <= 0.85):
        raise ValueError(f"Unexpected train/test ratio: {train_ratio:.3f}")

    classes = sorted(set(y_train.unique()) | set(y_test.unique()))
    if classes != [0, 1]:
        raise ValueError(f"Target classes are not binary [0,1]: {classes}")

    return (
        f"X_train={X_train.shape}, X_test={X_test.shape}, "
        f"class_balance_train={y_train.value_counts(normalize=True).to_dict()}"
    )


def verify_training_and_metrics(project_root: Path, min_test_accuracy: float) -> str:
    X_train, X_test, y_train, y_test, _ = load_prepare_split(
        resolve_data_path(project_root)
    )

    model = train_mlp_model(X_train, y_train)
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

    test_acc = metrics["test_accuracy"]
    if test_acc < min_test_accuracy:
        raise ValueError(
            f"Test accuracy {test_acc:.4f} is below required threshold {min_test_accuracy:.4f}"
        )

    return (
        f"train_accuracy={metrics['train_accuracy']:.4f}, "
        f"test_accuracy={metrics['test_accuracy']:.4f}"
    )


def verify_model_save_load_predict(project_root: Path) -> str:
    X_train, X_test, y_train, _, _ = load_prepare_split(resolve_data_path(project_root))

    model = train_mlp_model(X_train, y_train)

    with TemporaryDirectory() as tmp_dir:
        tmp_model_path = Path(tmp_dir) / "tmp_model.pkl"
        save_model(model, tmp_model_path)
        loaded = load_model(tmp_model_path)

        sample = X_test.head(1).copy()
        pred_class, pred_prob = predict_admission(loaded, sample)

    if pred_class not in (0, 1):
        raise ValueError(f"Invalid class prediction: {pred_class}")
    if not (0.0 <= pred_prob <= 1.0):
        raise ValueError(f"Invalid probability: {pred_prob}")

    return f"sample_prediction=class:{pred_class}, probability:{pred_prob:.4f}"


def verify_streamlit_wiring(project_root: Path) -> str:
    app_file = project_root / "streamlit_04NeuralNetworks_app.py"
    if not app_file.exists():
        app_file = project_root / "streamlit_app.py"

    content = app_file.read_text(encoding="utf-8")
    required_snippets = [
        "Admission.csv",
        "get_or_train_model",
        "predict_admission",
        "FEATURE_COLUMNS",
        'st.form("admission_form")',
    ]

    missing = [snippet for snippet in required_snippets if snippet not in content]
    if missing:
        raise ValueError(f"{app_file.name} missing snippets: {missing}")

    return f"{app_file.name} references data, model, and prediction flow correctly"


def verify_notebook_modular_section(project_root: Path) -> str:
    notebook_path = project_root / "04 UCLA_Neural_Networks_Solution.ipynb"
    if not notebook_path.exists():
        return "Notebook not found (optional for deployment)"

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])

    has_modular_header = any(
        "Modular Workflow (Streamlit Ready)" in "".join(cell.get("source", []))
        for cell in cells
        if cell.get("cell_type") == "markdown"
    )

    has_module_usage = any(
        "load_prepare_split" in "".join(cell.get("source", []))
        and "train_mlp_model" in "".join(cell.get("source", []))
        for cell in cells
        if cell.get("cell_type") == "code"
    )

    if not has_modular_header:
        raise ValueError("Notebook is missing the modular workflow markdown section")
    if not has_module_usage:
        raise ValueError("Notebook is missing modular training code usage")

    return "Notebook contains modular section and module-based training cells"


def verify_existing_artifacts(project_root: Path) -> str:
    model_path = project_root / "models" / "admission_mlp.pkl"
    confusion_path = project_root / "artifacts" / "confusion_matrix.png"

    existing = []
    if model_path.exists():
        existing.append("models/admission_mlp.pkl")
    if confusion_path.exists():
        existing.append("artifacts/confusion_matrix.png")

    if not existing:
        return "No prebuilt artifacts found (this is OK before running training)"

    return f"Existing artifacts: {', '.join(existing)}"


def print_report(results: list[CheckResult]) -> int:
    print("\n=== Verification Report ===")
    for idx, result in enumerate(results, start=1):
        status = "PASS" if result.passed else "FAIL"
        print(f"{idx:02d}. [{status}] {result.name}")
        print(f"    {result.details}")

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    print("\nSummary")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")

    return 0 if failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Project-wide verifier for UCLA admission app"
    )
    parser.add_argument(
        "--min-test-accuracy",
        type=float,
        default=0.90,
        help="Minimum acceptable test accuracy for training verification",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent

    checks = [
        run_check("Required files", lambda: verify_required_files(project_root)),
        run_check("Dataset schema", lambda: verify_dataset_schema(project_root)),
        run_check(
            "Data prep and split", lambda: verify_data_prep_and_split(project_root)
        ),
        run_check(
            "Training and evaluation",
            lambda: verify_training_and_metrics(project_root, args.min_test_accuracy),
        ),
        run_check(
            "Model save/load and inference",
            lambda: verify_model_save_load_predict(project_root),
        ),
        run_check("Streamlit wiring", lambda: verify_streamlit_wiring(project_root)),
        run_check(
            "Notebook modular section",
            lambda: verify_notebook_modular_section(project_root),
        ),
        run_check(
            "Existing artifacts", lambda: verify_existing_artifacts(project_root)
        ),
    ]

    return print_report(checks)


if __name__ == "__main__":
    raise SystemExit(main())
