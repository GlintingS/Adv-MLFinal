import logging

# Import mean absolute error
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)


# Function to predict and evaluate
def evaluate_model(model, X_test_scaled, y_test):
    logger.info("Evaluating model: %s", type(model).__name__)
    try:
        # make predictions on test set
        y_pred = model.predict(X_test_scaled)

        # Calculate the mean absolute error on test set
        test_mae = mean_absolute_error(y_pred, y_test)
        logger.info("Model MAE: %.4f", test_mae)

        return test_mae
    except Exception as exc:
        logger.error("Model evaluation failed: %s", exc)
        raise
