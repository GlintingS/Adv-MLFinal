from __future__ import annotations

import logging
import sys
from pathlib import Path
import warnings

from scr.Model.predict_models import evaluate_model
from scr.Model.train_models import save_model, train_mlp_model
from scr.data.make_dataset import load_prepare_split
from scr.visuals.visualize import plot_confusion_matrix, plot_loss_curve

warnings.filterwarnings(
    "ignore",
    message=".*encountered in matmul.*",
    category=RuntimeWarning,
)

logger = logging.getLogger(__name__)


def run_training_pipeline() -> None:
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "raw" / "Admission.csv"
    if not data_path.exists():
        data_path = project_root / "Admission.csv"
    model_path = project_root / "models" / "admission_mlp.pkl"
    confusion_path = project_root / "artifacts" / "confusion_matrix.png"
    loss_path = project_root / "artifacts" / "loss_curve.png"

    logger.info("Starting training pipeline")

    X_train, X_test, y_train, y_test, _ = load_prepare_split(data_path=data_path)

    model = train_mlp_model(X_train, y_train)
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    saved_path = save_model(model, model_path)

    plot_confusion_matrix(metrics["test_confusion_matrix"], confusion_path)

    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "loss_curve_"):
        plot_loss_curve(classifier.loss_curve_, loss_path)

    logger.info("Model saved to: %s", saved_path)
    logger.info("Train accuracy: %.4f", metrics["train_accuracy"])
    logger.info("Test accuracy:  %.4f", metrics["test_accuracy"])
    logger.info("Classification report:\n%s", metrics["classification_report"])

    print(f"Model saved to: {saved_path}")
    print(f"Train accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Test accuracy:  {metrics['test_accuracy']:.4f}")
    print("Classification report:\n")
    print(metrics["classification_report"])


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-36s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("training.log", encoding="utf-8"),
        ],
    )
    try:
        run_training_pipeline()
    except Exception:
        logger.exception("Training pipeline failed")
        sys.exit(1)
