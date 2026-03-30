# Import mean absolute error
from sklearn.metrics import mean_absolute_error


# Function to predict and evaluate
def evaluate_model(model, X_test_scaled, y_test):

    # make predictions on test set
    y_pred = model.predict(X_test_scaled)

    # Calculate the mean absolute error on test set
    test_mae = mean_absolute_error(y_pred, y_test)

    return test_mae
