import logging
import os
import time

import pandas as pd
import requests

from utils.utils import get_data_folder, pjoin

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_existing_data(filepath: str) -> pd.DataFrame:
    """Load cached activity data from CSV if it exists."""
    if os.path.isfile(filepath):
        logger.info(f"Loading cached data from {filepath}")
        return pd.read_csv(filepath)
    return pd.DataFrame()


def fetch_activities_from_api(
    target_id: str, base_url: str, request_limit: int, global_limit: int, delay: float
) -> list:
    """Fetch activity records from ChEMBL API."""
    all_activities = []
    offset = 0

    while offset < global_limit:
        params = {
            "target_chembl_id": target_id,
            "standard_type": "IC50",
            "limit": request_limit,
            "offset": offset,
            "format": "json",
        }

        response = requests.get(base_url, params=params)
        time.sleep(delay)
        response.raise_for_status()

        data = response.json()
        activities = data.get("activities", [])
        if not activities:
            break

        all_activities.extend(activities)
        logger.info(
            f"Fetched {len(activities)} new activities (Total: {len(all_activities)})"
        )
        offset += request_limit

    return all_activities


def activities_to_dataframe(activities: list) -> pd.DataFrame:
    """Convert a list of activities to a pandas DataFrame."""
    df = pd.DataFrame(activities)
    if "action_type" in df.columns:
        df["action_type"] = df["action_type"].apply(
            lambda x: x.get("action_type") if isinstance(x, dict) else x
        )
    return df


def parse_activities(
    target_id: str,
    base_url: str = "https://www.ebi.ac.uk/chembl/api/data/activity",
    request_limit: int = 1000,
    global_limit: int = 10000,
    delay: float = 0.05,
    cache_dir: str = None,
) -> pd.DataFrame:
    """Main function to fetch and optionally cache ChEMBL activity data."""
    cache_dir = cache_dir or get_data_folder()
    cache_path = pjoin(cache_dir, f"{target_id}_raw.csv")

    df = get_existing_data(cache_path)
    if not df.empty:
        return df

    logger.info(f"Fetching activities for target {target_id}...")
    activities = fetch_activities_from_api(
        target_id, base_url, request_limit, global_limit, delay
    )

    df = activities_to_dataframe(activities)
    df.to_csv(cache_path, index=False)
    logger.info(f"Saved {len(df)} records to {cache_path}")
    return df
