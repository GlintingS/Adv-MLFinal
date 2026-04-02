"""
verify.py — Pre-flight readiness check for the Mall Customer Segmentation project.

Run with:
    python verify.py

Each check prints PASS / FAIL with a short explanation. A final summary shows
how many checks passed so you know whether it is safe to run main.py or
launch the Streamlit app.
"""

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCR = ROOT / "scr"

# ── helpers ──────────────────────────────────────────────────────────────────

PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"
WARN = "\033[33m[WARN]\033[0m"
INFO = "\033[36m[INFO]\033[0m"

results = []


def check(label: str, ok: bool, detail: str = "", warn_only: bool = False):
    tag = (WARN if warn_only else FAIL) if not ok else PASS
    suffix = f"  → {detail}" if detail else ""
    print(f"  {tag}  {label}{suffix}")
    results.append((label, ok or warn_only))  # warnings count as passing


# ── 1. Python version ────────────────────────────────────────────────────────


def check_python():
    print("\n[1] Python version")
    major, minor = sys.version_info[:2]
    ok = major == 3 and minor >= 9
    check(
        f"Python {major}.{minor}",
        ok,
        "(requires 3.9+)" if not ok else "",
    )


# ── 2. Required packages ─────────────────────────────────────────────────────

REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "streamlit": "streamlit",
}


def check_packages():
    print("\n[2] Required packages")
    import subprocess

    for module, pip_name in REQUIRED_PACKAGES.items():
        # Use pip show to check installation without importing the module
        # (avoids accidental local-module shadowing of installed libraries).
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", pip_name],
            capture_output=True,
        )
        installed = result.returncode == 0
        check(pip_name, installed, f"pip install {pip_name}" if not installed else "")


# ── 3. Project structure ─────────────────────────────────────────────────────

REQUIRED_FILES = [
    ROOT / "main.py",
    ROOT / "streamlit_03UnsupervisedClustering_app.py",
    SCR / "data" / "__init__.py",
    SCR / "data" / "make_dataset.py",
    SCR / "Model" / "__init__.py",
    SCR / "Model" / "train_models.py",
    SCR / "Model" / "predict_models.py",
    SCR / "Model" / "hyperpara_tuning.py",
    SCR / "visuals" / "__init__.py",
    SCR / "visuals" / "visualize.py",
]

REQUIRED_DIRS = [
    ROOT / "data" / "raw",
    ROOT / "data" / "interim",
    ROOT / "data" / "processed",
    ROOT / "data" / "external",
]


def check_structure():
    print("\n[3] Project structure — files")
    for f in REQUIRED_FILES:
        check(str(f.relative_to(ROOT)), f.exists())

    print("\n[4] Project structure — data directories")
    for d in REQUIRED_DIRS:
        check(
            str(d.relative_to(ROOT)),
            d.exists(),
            "mkdir -p " + str(d.relative_to(ROOT)) if not d.exists() else "",
        )


# ── 4. Raw data file ─────────────────────────────────────────────────────────

RAW_CSV_CANDIDATES = [
    ROOT / "data" / "raw" / "mall_customers.csv",
    ROOT / "mall_customers.csv",
]

EXPECTED_COLUMNS = {"CustomerID", "Gender", "Age", "Annual_Income", "Spending_Score"}
EXPECTED_ROWS = 200


def check_data():
    print("\n[5] Raw data (mall_customers.csv)")

    csv_path = None
    for candidate in RAW_CSV_CANDIDATES:
        if candidate.exists():
            csv_path = candidate
            break

    check(
        "File found",
        csv_path is not None,
        "Place mall_customers.csv in data/raw/ or the project root",
    )

    if csv_path is None:
        return

    check(
        "Location (data/raw/)",
        csv_path == RAW_CSV_CANDIDATES[0],
        "File is at project root — consider moving it to data/raw/",
        warn_only=(csv_path != RAW_CSV_CANDIDATES[0]),
    )

    try:
        import pandas as pd

        df = pd.read_csv(csv_path)

        missing_cols = EXPECTED_COLUMNS - set(df.columns)
        check(
            "Expected columns present",
            not missing_cols,
            f"Missing: {missing_cols}" if missing_cols else "",
        )

        check(
            f"Row count (~{EXPECTED_ROWS})",
            len(df) > 0,
            f"Got {len(df)} rows",
        )

        null_counts = df.isnull().sum()
        has_nulls = null_counts.any()
        check(
            "No missing values",
            not has_nulls,
            str(null_counts[null_counts > 0].to_dict()) if has_nulls else "",
            warn_only=has_nulls,
        )

    except Exception as exc:
        # Cloud-only OneDrive files can't be read until synced locally.
        cloud_hint = (
            " — sync the file first (OneDrive: right-click → Keep on this device)"
            if any(
                kw in str(exc).lower()
                for kw in ("timed out", "error reading", "no columns")
            )
            else ""
        )
        check("CSV readable", False, str(exc) + cloud_hint, warn_only=True)


# ── 5. Module imports ────────────────────────────────────────────────────────


def check_imports():
    print("\n[6] Module imports (scr/)")
    sys.path.insert(0, str(SCR))

    modules = {
        "make_dataset": "data.make_dataset",
        "train_models": "Model.train_models",
        "predict_models": "Model.predict_models",
        "hyperpara": "Model.hyperpara_tuning",
        "visualize": "visuals.visualize",
    }

    for label, module_path in modules.items():
        try:
            importlib.import_module(module_path)
            check(module_path, True)
        except Exception as exc:
            check(module_path, False, str(exc))


# ── 6. Saved model (optional) ────────────────────────────────────────────────

MODEL_PATH = ROOT / "models" / "kmodel.pkl"


def check_model():
    print("\n[7] Saved model (optional)")
    if MODEL_PATH.exists():
        try:
            import pickle

            with MODEL_PATH.open("rb") as f:
                mdl = pickle.load(f)
            check("models/kmodel.pkl loadable", True, f"n_clusters={mdl.n_clusters}")
        except Exception as exc:
            check("models/kmodel.pkl loadable", False, str(exc))
    else:
        check(
            "models/kmodel.pkl",
            True,
            "Not yet trained — run main.py or train inside the app",
            warn_only=True,
        )


# ── 7. Streamlit reachability ────────────────────────────────────────────────


def check_streamlit():
    print("\n[8] Streamlit entry point")
    sl = ROOT / "streamlit_03UnsupervisedClustering_app.py"
    check("streamlit_03UnsupervisedClustering_app.py exists", sl.exists())
    if sl.exists():
        src = sl.read_text()
        check("st.set_page_config present", "set_page_config" in src)
        check("st.cache_data present", "cache_data" in src)


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Mall Customer Segmentation — Readiness Check")
    print("=" * 60)

    check_python()
    check_packages()
    check_structure()
    check_data()
    check_imports()
    check_model()
    check_streamlit()

    total = len(results)
    passed = sum(ok for _, ok in results)
    failed = total - passed

    print("\n" + "=" * 60)
    print(f"  Result: {passed}/{total} checks passed", end="")
    if failed:
        print(f"  |  {failed} issue(s) need attention")
    else:
        print("  — all good!")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
