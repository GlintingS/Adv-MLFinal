# Real Estate Price Predictor

A machine learning project that predicts real estate prices using Linear Regression and Random Forest models. Includes a Streamlit web application for interactive predictions.

## Project Structure

```
ML2-TP01RealEstate/
├── main.py                          # Main pipeline: load data, train, evaluate, plot
├── streamlit_01RealEstate_app.py    # Streamlit web app for interactive predictions
├── verify.py                        # Project verification/validation script
├── requirements.txt                 # Python dependencies
├── data/
│   ├── raw/
│   │   └── real_estate.csv          # Original dataset
│   └── processed/
│       └── cleaned_data.csv         # Preprocessed dataset (generated)
├── models/                          # Trained model files (.pkl, generated)
├── scr/
│   ├── data/
│   │   └── make_dataset.py          # Data loading and preprocessing
│   ├── Model/
│   │   ├── train_models.py          # Model training (LR and RF)
│   │   └── predict_models.py        # Model evaluation
│   └── visuals/
│       └── visualize.py             # MAE comparison bar chart
```

## Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`:
  - **streamlit** — Web application framework
  - **pandas** — Data manipulation
  - **scikit-learn** — Machine learning models and metrics
  - **matplotlib** — Plotting and visualization

## Installation

1. Clone or download this repository.
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   .venv\Scripts\activate      # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training Pipeline

Run the full pipeline (preprocess data, train models, evaluate, and generate the MAE comparison plot):

```bash
python main.py
```

This will:
- Load and preprocess `data/raw/real_estate.csv`
- Save cleaned data to `data/processed/cleaned_data.csv`
- Train a Linear Regression and a Random Forest model
- Save trained models to the `models/` directory (`LRmodel.pkl`, `RFmodel.pkl`)
- Evaluate both models and display a MAE comparison chart
- Write logs to `pipeline.log`

### Streamlit Web App

Launch the interactive price prediction app:

```bash
streamlit run streamlit_01RealEstate_app.py
```

The app allows you to:
- Enter property details (tax, insurance, sqft, lot size, age) and get a predicted price
- View feature importance for the Random Forest model
- Compare model evaluation metrics (MAE, RMSE, R²)
- Explore market insights with price distribution and scatter plots

> **Note:** The app expects `models/RFmodel.pkl` to exist. Run the training pipeline first, or upload a model file through the app.

## Logging

All modules use Python's built-in `logging` library. When running the training pipeline, logs are written to both the console and `pipeline.log`. Log messages cover data loading, model training, evaluation, and any errors encountered.

## Author

Frank Song
