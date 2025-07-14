import csv
import json
import os
import time
import urllib.parse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from utils.utils import get_data_folder, pjoin

# ChEMBL API URL
CHEMBL_URL = "https://www.ebi.ac.uk/chembl/api/data/"
# GET request limit set in ChEMBL
CHEMBL_LIMIT = 1000
# max elements to retrieve
GLOBAL_LIMIT = 10000


def parse_activities(target_id: str) -> pd.DataFrame:
    path = pjoin(get_data_folder(), f"{target_id}_raw.csv")

    # read dataset file if exists
    if os.path.isfile(path):
        return pd.read_csv(path)

    all_activities = []
    offset = 0

    # multiple tries to retrieve all (limited by global_limit) activities
    while offset < GLOBAL_LIMIT:
        activities_url = (
            f""
            f"{CHEMBL_URL}activity?target_chembl_id={target_id}&"
            f"standard_type=IC50&"
            f"limit={CHEMBL_LIMIT}&"
            f"offset={offset}"
            f""
        )
        response = requests.get(activities_url, params={"format": "json"})

        # pause between requests
        time.sleep(0.05)
        response.raise_for_status()

        data = json.loads(response.content)
        activities = data.get("activities", [])
        if not activities:
            break

        all_activities.extend(activities)
        print(
            f"Downloaded {len(activities)} activities for {target_id} (Total: {len(all_activities)})"
        )
        offset += CHEMBL_LIMIT

    # create DF from data
    df_activities = []
    for activity in all_activities:
        df = pd.Series(activity).to_frame().T
        df_activities.append(df)

    df_activities = pd.concat(df_activities, ignore_index=True)

    df_activities["action_type"] = [
        None if i is None else i["action_type"] for i in df_activities["action_type"]
    ]

    # save dataset to csv
    df_activities.to_csv(path, index=False)
    return df_activities


if __name__ == "__main__":
    parse_activities(
        "CHEMBL2760"
    )  # Example target ID, replace with actual target ID as needed
