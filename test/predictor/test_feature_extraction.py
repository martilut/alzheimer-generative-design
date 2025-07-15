import os

import pandas as pd
import pytest

from agd.predictor.feature_extraction import (
    get_features_dataset,
    smiles_to_features,
)
from utils.utils import get_project_path, pjoin

# Constants
DATA_DIR = pjoin(get_project_path(), "test", "resources")
TARGET_ID = "CHEMBL262"


def test_smiles_to_features_single_rdkit():
    df = pd.read_csv(os.path.join(DATA_DIR, f"{TARGET_ID}_clean.csv"), index_col=0)
    smiles = df["canonical_smiles"].dropna().iloc[0]

    features = smiles_to_features(smiles, descriptors=["rdkit"])
    assert isinstance(features, pd.DataFrame)
    assert features.shape[0] == 1


def test_smiles_to_features_with_fingerprint():
    df = pd.read_csv(os.path.join(DATA_DIR, f"{TARGET_ID}_clean.csv"), index_col=0)
    smiles = df["canonical_smiles"].dropna().iloc[0]

    fingerprints = {"morgan": {"params": {"radius": 2, "fpSize": 16}}}

    features = smiles_to_features(smiles, fingerprints=fingerprints)
    assert isinstance(features, pd.DataFrame)
    assert features.shape[0] == 1
    assert len(features.columns) == 16


def test_get_features_dataset_fingerprint_only():
    fingerprints = {"morgan": {"params": {"radius": 2, "fpSize": 4}}}

    df = get_features_dataset(
        TARGET_ID,
        descriptors=None,
        fingerprints=fingerprints,
        rewrite=True,
        data_dir=DATA_DIR,
    )

    assert isinstance(df, pd.DataFrame)
    assert "smiles" in df.columns
    assert "standard_value" in df.columns
    assert df.index.name == "molecule_chembl_id"
    assert not df.empty
    assert df.shape[1] == 6  # 4 fingerprint columns + 2 additional columns


def test_get_features_dataset_all_features():
    fingerprints = {"morgan": {"params": {"radius": 2, "fpSize": 4}}}

    df = get_features_dataset(
        TARGET_ID,
        descriptors=["rdkit"],
        fingerprints=fingerprints,
        rewrite=True,
        data_dir=DATA_DIR,
    )

    assert isinstance(df, pd.DataFrame)
    assert "smiles" in df.columns
    assert "standard_value" in df.columns
    assert df.index.name == "molecule_chembl_id"
    assert not df.empty
    assert df.shape[1] > 6
