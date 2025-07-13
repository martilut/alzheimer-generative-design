import numpy as np
import pandas as pd


def remove_collinear_features(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
  corr_matrix = df.corr().abs()
  upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
  to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
  return df.drop(columns=to_drop)
