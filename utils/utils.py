import os
from os.path import join
from pathlib import Path


def get_project_path() -> str:
    return str(Path(__file__).parent.parent)


def pjoin(*args) -> Path:
    return Path(join(*args))


def get_data_folder() -> str:
    return pjoin(get_project_path(), "data")
