# UCLA Admission Chance Classifier — Neural Networks

A machine-learning project that trains an **MLPClassifier** (multi-layer perceptron) to predict whether a student is likely to be admitted to UCLA based on profile features such as GRE score, TOEFL score, CGPA, and more.

## Purpose

The project demonstrates an end-to-end ML workflow:

1. **Data loading & preparation** — reads `Admission.csv`, engineers a binary target from the continuous admission-chance column, and splits into train/test sets.
2. **Model training** — builds a scikit-learn `Pipeline` (MinMaxScaler + OneHotEncoder → MLPClassifier) and fits it on the training data.
3. **Evaluation** — reports accuracy, a classification report, and a confusion matrix.
4. **Prediction** — exposes a function (and a Streamlit UI) for single-sample inference.
5. **Hyperparameter tuning** — optional grid search over hidden-layer sizes, activations, and regularisation strength.

## Project Structure

```
├── main.py                              # CLI training pipeline
├── streamlit_04NeuralNetworks_app.py    # Streamlit web application
├── streamlit_app.py                     # Streamlit Cloud entrypoint
├── verify.py                            # Automated verification suite
├── requirements.txt                     # Python dependencies
├── data/
│   └── raw/
│       └── Admission.csv                # Source dataset
├── models/                              # Saved model artifacts (.pkl)
├── artifacts/                           # Generated plots (confusion matrix, loss curve)
└── scr/
    ├── data/
    │   └── make_dataset.py              # Data loading, feature prep, train/test split
    ├── Model/
    │   ├── train_models.py              # Pipeline construction, training, save/load
    │   ├── predict_models.py            # Evaluation metrics and single-sample prediction
    │   └── hyperpara_tuning.py          # Grid-search hyperparameter tuning
    └── visuals/
        └── visualize.py                 # Confusion matrix and loss-curve plotting
```

## Requirements

- **Python 3.10+**
- Dependencies listed in `requirements.txt`:

| Package | Minimum Version |
|---------|----------------|
| numpy | 1.24 |
| pandas | 2.0 |
| scikit-learn | 1.3 |
| matplotlib | 3.7 |
| seaborn | 0.12 |
| streamlit | 1.30 |

## Installation

```bash
# Clone or download the repository, then:
cd ML2-TP04NeuralNetworks
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

## How to Run

### Train the model (CLI)

```bash
python main.py
```

This loads the dataset, trains the MLP model, saves the model to `models/`, and writes evaluation artifacts to `artifacts/`. Training logs are also written to `training.log`.

### Launch the Streamlit app

```bash
streamlit run streamlit_app.py
```

The web app lets you adjust the admission threshold, retrain the model interactively, enter student profile data, and view predictions along with attribute-importance charts.

### Run the verification suite

```bash
python verify.py
```

Runs automated checks covering file presence, dataset schema, data splitting, model training accuracy, save/load round-tripping, and Streamlit wiring.

## Logging

All modules use Python's built-in `logging` library. When running via `main.py`, logs are emitted to both **stdout** and a `training.log` file. The Streamlit app logs to stdout. Log messages cover data loading, training progress, model persistence, evaluation results, and any errors encountered.
