from pathlib import Path

# setup(
#     name='ML2 Final Real Estate',
#     packages=  ,
#     version='1.0.0',
#     description='',
#     author='Frank Song',
#     license='',
# )

from scr.data import make_dataset
from scr.Model import predict_models, train_models
from scr.visuals import visualize

if __name__ == "__main__":

    # Load and the raw data
    root = Path(__file__).resolve().parent
    data_path = root / "data" / "raw" / "real_estate.csv"

    # Preprocess the raw data
    df, X, y = make_dataset.load_and_preprocess_data(str(data_path))

    # Train the linear regression model
    LRmodel, X_LR_scaled, y_LR_test = train_models.train_LRmodel(X, y)

    # Train the Random Forest model
    RFmodel, X_RF_test, y_RF_test = train_models.train_RFmodel(X, y)

    # Evaluate the model
    LR_mae = predict_models.evaluate_model(LRmodel, X_LR_scaled, y_LR_test)
    RF_mae = predict_models.evaluate_model(RFmodel, X_RF_test, y_RF_test)

    # plot the mae values for both models
    models = ["Linear Regression", "Random Forest"]
    mae_values = [LR_mae, RF_mae]

    visualize.plot_mae(models, mae_values)
