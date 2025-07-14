import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, cross_validate
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

from utils.utils import flatten_pipeline_info, plot_predictions


def get_preprocessing_pipeline(selector=None, scaler=None, dim_reducer=None):
    steps = []
    if selector:
        steps.append(('feature_selection', selector))
    if scaler:
        steps.append(('scaler', scaler))
    if dim_reducer:
        steps.append(('dim_reduction', dim_reducer))
    if len(steps) == 0:
      return None
    return Pipeline(steps)

def get_model_pipeline(preprocessing_pipeline, model):
    model_pipeline = []
    if preprocessing_pipeline is not None:
      model_pipeline.append(('preprocessing', preprocessing_pipeline))
    model_pipeline.append(('model', model))
    return Pipeline(model_pipeline)

def run_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    model,
    selector=None,
    scaler=None,
    dim_reducer=None,
    scoring: list = [
        'r2',
        'neg_root_mean_squared_error',
        'neg_mean_absolute_error'
    ],
    results_path: str = 'results.csv'
):
    if isinstance(y, pd.DataFrame):
        y = y.squeeze()

    preprocessing = get_preprocessing_pipeline(
        selector=selector,
        scaler=scaler,
        dim_reducer=dim_reducer
    )
    pipeline = get_model_pipeline(preprocessing, model)

    scores = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        return_estimator=False,
    )

    # Print scores
    print("Cross-validation mean scores:")
    for metric in scoring:
        mean_score = np.mean(scores[f'test_{metric}'])
        std_score = np.std(scores[f'test_{metric}'])
        print(f"  {metric}, mean: {mean_score:.4f}, std: {std_score:.4f}")

    # Predictions
    y_pred = cross_val_predict(pipeline, X, y, cv=cv)
    plot_predictions(y, y_pred)

    # Score dictionary
    score_data = {}
    for metric in scoring:
        vals = scores[f'test_{metric}']
        score_data[f"{metric}_mean"] = np.mean(vals)
        for i, val in enumerate(vals, start=1):
            score_data[f"{metric}_fold{i}"] = val

    # Timing
    fit_times = scores['fit_time']
    score_times = scores['score_time']
    score_data['fit_time_mean'] = np.mean(fit_times)
    score_data['score_time_mean'] = np.mean(score_times)
    for i, val in enumerate(fit_times, start=1):
        score_data[f"fit_time_fold{i}"] = val
    for i, val in enumerate(score_times, start=1):
        score_data[f"score_time_fold{i}"] = val

    # Flatten pipeline
    pipeline_info = flatten_pipeline_info(pipeline)
    combined_dict = {**pipeline_info, **score_data}
    new_row_df = pd.DataFrame([combined_dict])

    # Append or skip based on duplicate preprocessing + model config
    check_cols = ['preprocessing_params', 'model_class', 'model_params']

    if os.path.isfile(results_path):
        df_existing = pd.read_csv(results_path)
        if 'preprocessing_params' not in list(new_row_df.columns):
            check_cols = check_cols[1:]
        is_duplicate = (
            (df_existing[check_cols] == new_row_df[check_cols].iloc[0]).all(axis=1)
        ).any()

        if is_duplicate:
            print("Duplicate experiment (same preprocessing + model config). Skipping.")
        else:
            df_all = pd.concat([df_existing, new_row_df], ignore_index=True)
            df_all.to_csv(results_path, index=False)
            print(f"Appended new experiment to {results_path}")
    else:
        new_row_df.to_csv(results_path, index=False)
        print(f"Created new results file at {results_path}")

    return pipeline
