import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import (
    KFold,
    cross_val_predict,
    cross_val_score,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.utils import flatten_pipeline_info, plot_predictions


def get_preprocessing_pipeline(selector=None, scaler=None, dim_reducer=None):
    steps = []
    if selector:
        steps.append(("feature_selection", selector))
    if scaler:
        steps.append(("scaler", scaler))
    if dim_reducer:
        steps.append(("dim_reduction", dim_reducer))
    if len(steps) == 0:
        return None
    return Pipeline(steps)


def get_model_pipeline(preprocessing_pipeline, model):
    model_pipeline = []
    if preprocessing_pipeline is not None:
        model_pipeline.append(("preprocessing", preprocessing_pipeline))
    model_pipeline.append(("model", model))
    return Pipeline(model_pipeline)


def run_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    model,
    selector=None,
    scaler=None,
    dim_reducer=None,
    scoring: list = ["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"],
    results_path: str = "results.csv",
):
    if isinstance(y, pd.DataFrame):
        y = y.squeeze()

    # Construct pipeline
    preprocessing = get_preprocessing_pipeline(
        selector=selector, scaler=scaler, dim_reducer=dim_reducer
    )
    pipeline = get_model_pipeline(preprocessing, model)

    # Single pass cross-validation: get scores and fitted estimators
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        return_estimator=True,
        n_jobs=-1,
        verbose=1,  # prints progress info
    )

    # Use fitted estimators to generate predictions faster
    y_pred = np.zeros_like(y, dtype=float)

    for (train_idx, test_idx), est in zip(cv.split(X, y), scores["estimator"]):
        X_test = X.iloc[test_idx]
        y_pred[test_idx] = est.predict(X_test)

    # Plot predictions once
    plot_predictions(y, y_pred)

    # Print cross-validation scores
    print("Cross-validation mean scores:")
    for metric in scoring:
        mean_score = np.mean(scores[f"test_{metric}"])
        std_score = np.std(scores[f"test_{metric}"])
        print(f"  {metric}, mean: {mean_score:.4f}, std: {std_score:.4f}")

    # Compile score data
    score_data = {
        f"{metric}_mean": np.mean(scores[f"test_{metric}"]) for metric in scoring
    }
    for metric in scoring:
        for i, val in enumerate(scores[f"test_{metric}"], start=1):
            score_data[f"{metric}_fold{i}"] = val

    # Timing
    score_data["fit_time_mean"] = np.mean(scores["fit_time"])
    score_data["score_time_mean"] = np.mean(scores["score_time"])
    for i, val in enumerate(scores["fit_time"], start=1):
        score_data[f"fit_time_fold{i}"] = val
    for i, val in enumerate(scores["score_time"], start=1):
        score_data[f"score_time_fold{i}"] = val

    # Flatten pipeline info for logging
    pipeline_info = flatten_pipeline_info(pipeline)
    combined_dict = {**pipeline_info, **score_data}
    new_row_df = pd.DataFrame([combined_dict])

    # Append if not duplicate
    check_cols = ["preprocessing_params", "model_class", "model_params"]
    if os.path.isfile(results_path):
        df_existing = pd.read_csv(results_path)
        df_all = pd.concat([df_existing, new_row_df], ignore_index=True)
        df_all.to_csv(results_path, index=False)
        print(f"Appended new experiment to {results_path}")
    else:
        new_row_df.to_csv(results_path, index=False)
        print(f"Created new results file at {results_path}")

    # Return last fitted estimator
    return scores["estimator"][-1], pipeline  # or return all estimators if needed
