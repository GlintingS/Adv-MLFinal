import logging
import sys
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        # Load and the raw data
        root = Path(__file__).resolve().parent
        data_path = root / "data" / "raw" / "real_estate.csv"

        # Preprocess the raw data
        logger.info("Starting data preprocessing")
        df, X, y = make_dataset.load_and_preprocess_data(str(data_path))

        # Train the linear regression model
        logger.info("Starting model training")
        LRmodel, X_LR_scaled, y_LR_test = train_models.train_LRmodel(X, y)

        # Train the Random Forest model
        RFmodel, X_RF_test, y_RF_test = train_models.train_RFmodel(X, y)

        # Evaluate the model
        logger.info("Starting model evaluation")
        LR_mae = predict_models.evaluate_model(LRmodel, X_LR_scaled, y_LR_test)
        RF_mae = predict_models.evaluate_model(RFmodel, X_RF_test, y_RF_test)

        # plot the mae values for both models
        models = ["Linear Regression", "Random Forest"]
        mae_values = [LR_mae, RF_mae]

        visualize.plot_mae(models, mae_values)
        logger.info("Pipeline completed successfully")

    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("Data error: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Pipeline failed with unexpected error: %s", exc, exc_info=True)
        sys.exit(1)
