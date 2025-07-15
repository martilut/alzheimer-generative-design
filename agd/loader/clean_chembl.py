from pathlib import Path
from typing import List, Optional

import pandas as pd

from agd.utils.utils import get_data_folder, pjoin

FILTER_COLUMNS = [
    "action_type",
    "activity_comment",
    "assay_type",
    "bao_label",
    "canonical_smiles",
    "data_validity_comment",
    "pchembl_value",
    "potential_duplicate",
    "standard_flag",
    "standard_relation",
    "standard_units",
    "standard_value",
    "type",
    "upper_value",
    "value",
    "molecule_chembl_id",
]

NUMERIC_COLUMNS = [
    "pchembl_value",
    "potential_duplicate",
    "standard_flag",
    "standard_value",
    "upper_value",
    "value",
]


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[FILTER_COLUMNS].dropna(subset=["standard_value"]).drop_duplicates()
    df[NUMERIC_COLUMNS] = (
        df[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce").astype(float)
    )
    return df.where(pd.notna(df), None)


def safe_filter(df: pd.DataFrame, col: str, value) -> pd.DataFrame:
    filtered = df[df[col] == value]
    return filtered if not filtered.empty else df


def safe_dropna(df: pd.DataFrame, subset: List[str]) -> pd.DataFrame:
    dropped = df.dropna(subset=subset)
    return dropped if not dropped.empty else df


def handle_duplicates(df: pd.DataFrame, molecule_id: str) -> Optional[pd.DataFrame]:
    df = df[df["molecule_chembl_id"] == molecule_id]
    df = df[
        (df["type"] == "IC50")
        & (df["standard_units"] == "nM")
        & df["canonical_smiles"].notna()
    ]

    if df.empty:
        return None

    df = safe_filter(df, "potential_duplicate", 0)
    df = safe_filter(df, "standard_flag", 1)
    df = safe_filter(df, "standard_relation", "=")
    df = safe_dropna(df, ["action_type"])

    df["data_validity_comment"] = [
        "valid" if x is None else None for x in df["data_validity_comment"]
    ]
    df = safe_dropna(df, ["data_validity_comment"])

    if len(df) > 1:
        median_val = df["standard_value"].median()
        closest_idx = (df["standard_value"] - median_val).abs().idxmin()
        return df.loc[[closest_idx]]

    return df


def remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    return df.drop(columns=constant_cols)


def clean_data(target_id: str, rewrite: bool = False) -> pd.DataFrame:
    raw_path = Path(pjoin(get_data_folder(), f"{target_id}_raw.csv"))
    clean_path = Path(pjoin(get_data_folder(), f"{target_id}_clean.csv"))

    if clean_path.exists() and not rewrite:
        return pd.read_csv(clean_path, index_col=0)

    df = pd.read_csv(raw_path)
    df_clean = filter_data(df)

    unique_dfs = []
    for mol_id, count in df_clean["molecule_chembl_id"].value_counts().items():
        df_mol = handle_duplicates(df_clean, mol_id)
        if df_mol is not None and len(df_mol) == 1:
            unique_dfs.append(df_mol)

    df_final = pd.concat(unique_dfs, ignore_index=True)
    df_final = remove_constant_columns(df_final)

    if df_final["molecule_chembl_id"].duplicated().any():
        raise ValueError("Duplicate molecule IDs detected after filtering.")

    df_final.set_index("molecule_chembl_id", inplace=True)
    df_final = df_final[["canonical_smiles", "standard_value"]]
    df_final.to_csv(clean_path)

    return df_final
