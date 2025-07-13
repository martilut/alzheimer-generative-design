import os

import pandas as pd

from utils.utils import pjoin, get_data_folder


# initial data filtering
def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()

    # leave columns relevant for filtering
    df_clean = df_clean.loc[
        :,
        [
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
        ],
    ]

    # drop missing and complete duplicate values
    df_clean.dropna(subset=["standard_value"], axis=0, inplace=True)
    df_clean.drop_duplicates(inplace=True)

    def convert_to_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
        for col in cols:
            df[col] = pd.to_numeric(df[col])
        return df

    # convert numeric columns
    df_clean = convert_to_numeric(
        df_clean,
        [
            "pchembl_value",
            "potential_duplicate",
            "standard_flag",
            "standard_value",
            "upper_value",
            "value",
        ],
    )

    # replace np.nan with None in non-numeric columns
    df_clean = df_clean.where(pd.notna(df_clean), None)

    return df_clean


# filter by criteria or skip if filter returns zero values
def safe_filter(df: pd.DataFrame, col: str, value: str) -> pd.DataFrame:
    df_filtered = df[df[col] == value]
    if df_filtered.shape[0] == 0:
        return df
    else:
        return df_filtered


# drop by criteria or skip
def safe_dropna(df: pd.DataFrame, subset: list) -> pd.DataFrame:
    df_dropped = df.dropna(subset=subset)
    if df_dropped.shape[0] == 0:
        return df
    else:
        return df_dropped


# choose one entry for duplicating molecule id
def handle_duplicates(df: pd.DataFrame, molecule_id: str) -> pd.DataFrame | None:
    # strict filters
    df_unique = df[(df["molecule_chembl_id"] == molecule_id)]
    df_unique = df_unique[(df_unique["type"] == "IC50")]
    df_unique = df_unique[(df_unique["standard_units"] == "nM")]
    df_unique.dropna(subset=["canonical_smiles"])

    if df_unique.shape[0] == 0:
        return None

    # soft filters
    df_unique = safe_filter(df_unique, "potential_duplicate", 0)
    df_unique = safe_filter(df_unique, "assay_type", "B")
    df_unique = safe_filter(df_unique, "bao_label", "single protein format")
    df_unique = safe_filter(df_unique, "standard_flag", 1)
    df_unique = safe_filter(df_unique, "standard_relation", "=")
    df_unique = safe_dropna(df_unique, ["action_type"])

    df_unique["data_validity_comment"] = [
        "valid" if i is None else None for i in df_unique["data_validity_comment"]
    ]
    df_unique = safe_dropna(df_unique, ["data_validity_comment"])

    # take entry by median standard_value if there are still multiple entries
    if df_unique.shape[0] > 1:
        standard_median = df_unique["standard_value"].median()
        idx = (df_unique["standard_value"] - standard_median).abs().idxmin()
        return df_unique.loc[idx, :].to_frame().T

    return df_unique


def remove_constant_columns(df: pd.DataFrame) -> None:
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    df.drop(columns=constant_cols, inplace=True)


# parse all unique molecule ids, handle duplicate entries
def clean_data(target_id: str, rewrite: bool = False) -> pd.DataFrame:
    df = pd.read_csv(pjoin(get_data_folder(), f"{target_id}_raw.csv"))
    path = pjoin(get_data_folder(), f"{target_id}_clean.csv")

    # read filtered dataset file if exists
    if os.path.exists(path) and not rewrite:
        return pd.read_csv(path, index_col=0)

    df_clean = filter_data(df)
    df_unique = []
    counter = df_clean["molecule_chembl_id"].value_counts()

    # for each molecule leave 1 or 0 entries
    for molecule_id in counter.index:
        df_molecule = handle_duplicates(df_clean, molecule_id)
        if df_molecule is not None and df_molecule.shape[0] == 1:
            df_unique.append(df_molecule)
        else:
            print(f"Removed all ({counter[molecule_id]}) entries for {molecule_id}")

    df_unique = pd.concat(df_unique)
    remove_constant_columns(df_unique)

    # check if IDs are unique
    molecules_unique = df_unique["molecule_chembl_id"].values
    if len(molecules_unique) != len(set(molecules_unique)):
        raise ValueError(
            f"Duplicate IDs: {len(molecules_unique)}, {len(set(molecules_unique))}"
        )

    # save filtered dataset to csv
    df_unique.set_index("molecule_chembl_id", drop=True, inplace=True)
    df_unique = df_unique.loc[:, ["canonical_smiles", "standard_value"]]
    df_unique.to_csv(path, index=True)

    return df_unique


if __name__ == "__main__":
    df_cleaned = clean_data("CHEMBL230", rewrite=True)
    print(f"Filtered dataset shape: {df_cleaned.shape}")
