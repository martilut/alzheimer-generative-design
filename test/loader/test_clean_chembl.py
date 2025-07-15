import pandas as pd
import pytest

from agd.loader.clean_chembl import (
    filter_data,
    safe_filter,
    safe_dropna,
    handle_duplicates,
)


# Sample minimal DataFrame
@pytest.fixture
def raw_data():
    return pd.DataFrame([
        {
            "action_type": "INHIBITOR",
            "activity_comment": None,
            "assay_type": "B",
            "bao_label": "single protein format",
            "canonical_smiles": "CCO",
            "data_validity_comment": None,
            "pchembl_value": "7.0",
            "potential_duplicate": "0",
            "standard_flag": "1",
            "standard_relation": "=",
            "standard_units": "nM",
            "standard_value": "50",
            "type": "IC50",
            "upper_value": "60",
            "value": "55",
            "molecule_chembl_id": "CHEMBL123"
        }
    ])

def test_filter_data(raw_data):
    df_filtered = filter_data(raw_data)
    assert df_filtered.shape[0] == 1
    assert df_filtered["standard_value"].dtype == float
    assert df_filtered["action_type"].iloc[0] == "INHIBITOR"

def test_safe_filter_keeps_original_if_empty(raw_data):
    df = safe_filter(raw_data, "standard_flag", "nonexistent")
    assert df.equals(raw_data)

def test_safe_dropna_keeps_original_if_all_nan(raw_data):
    raw_data["action_type"] = None
    df = safe_dropna(raw_data, ["action_type"])
    assert df.equals(raw_data)

def test_handle_duplicates_returns_clean_entry(raw_data):
    df = filter_data(raw_data)
    result = handle_duplicates(df, "CHEMBL123")
    assert result is not None
    assert result.shape[0] == 1
    assert result["molecule_chembl_id"].iloc[0] == "CHEMBL123"
