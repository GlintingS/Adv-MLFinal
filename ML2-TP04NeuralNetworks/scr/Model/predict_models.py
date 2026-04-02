from __future__ import annotations

import logging

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate classifier on train/test data and return key metrics."""
    logger.info(
        "Evaluating model on train (%d) and test (%d) samples",
        len(X_train),
        len(X_test),
    )
    try:
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
    except Exception:
        logger.exception("Model prediction failed during evaluation")
        raise

    results = {
        "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
        "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
        "test_confusion_matrix": confusion_matrix(y_test, y_test_pred).tolist(),
        "classification_report": classification_report(y_test, y_test_pred),
    }
    logger.info(
        "Train accuracy: %.4f, Test accuracy: %.4f",
        results["train_accuracy"],
        results["test_accuracy"],
    )
    return results


def predict_admission(model, input_frame: pd.DataFrame) -> tuple[int, float]:
    """Predict class and probability for one or more samples."""
    try:
        predicted_class = int(model.predict(input_frame)[0])
        probability = float(model.predict_proba(input_frame)[0][1])
    except Exception:
        logger.exception(
            "Prediction failed for input frame with shape %s", input_frame.shape
        )
        raise
    logger.info("Prediction: class=%d, probability=%.4f", predicted_class, probability)
    return predicted_class, probability
