from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GridSearchCV

from scr.Model.train_models import build_mlp_pipeline


def tune_mlp_model(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    """Run a small grid search to tune MLP hyperparameters."""
    pipeline = build_mlp_pipeline(X_train)

    param_grid = {
        "classifier__hidden_layer_sizes": [(6,), (8, 4), (12, 6)],
        "classifier__activation": ["relu", "tanh"],
        "classifier__alpha": [0.0001, 0.001, 0.01],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search
