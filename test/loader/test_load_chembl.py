import os
import tempfile
from unittest import mock

import pandas as pd
import pytest

from agd.loader.load_chembl import get_existing_data, activities_to_dataframe, fetch_activities_from_api


@pytest.fixture
def dummy_activity_data():
    return [
        {
            "activity_id": 1,
            "molecule_chembl_id": "CHEMBL123",
            "standard_value": "50",
            "action_type": {"action_type": "INHIBITOR"},
        },
        {
            "activity_id": 2,
            "molecule_chembl_id": "CHEMBL456",
            "standard_value": "100",
            "action_type": None,
        },
    ]


def test_activities_to_dataframe(dummy_activity_data):
    df = activities_to_dataframe(dummy_activity_data)
    assert isinstance(df, pd.DataFrame)
    assert "molecule_chembl_id" in df.columns
    assert df["action_type"].iloc[0] == "INHIBITOR"
    assert df["action_type"].iloc[1] is None


def test_get_existing_data_reads_csv():
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as f:
        test_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        test_df.to_csv(f.name, index=False)
        f.seek(0)

    df = get_existing_data(f.name)
    assert df.equals(test_df)

    os.remove(f.name)


@mock.patch("requests.get")
def test_fetch_activities_from_api(mock_get, dummy_activity_data):
    mock_response_with_data = mock.Mock()
    mock_response_with_data.status_code = 200
    mock_response_with_data.json.return_value = {"activities": dummy_activity_data}

    mock_response_empty = mock.Mock()
    mock_response_empty.status_code = 200
    mock_response_empty.json.return_value = {"activities": []}

    # First call returns data, second call returns empty to stop pagination
    mock_get.side_effect = [mock_response_with_data, mock_response_empty]

    results = fetch_activities_from_api(
        target_id="CHEMBL999",
        base_url="https://mock.api",
        request_limit=100,
        global_limit=200,
        delay=0.01
    )

    assert len(results) == len(dummy_activity_data)

