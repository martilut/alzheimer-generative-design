import os
from os.path import join
from pathlib import Path

from rdkit import Chem
import matplotlib.pyplot as plt
import seaborn as sns
import json


def get_project_path() -> str:
    return str(Path(__file__).parent.parent)


def pjoin(*args) -> Path:
    return Path(join(*args))


def get_data_folder() -> str:
    return pjoin(get_project_path(), "data")


def get_molecule(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as e:
        mol = None
    return mol


# plot predictions
def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    plt.plot(
        [min(y_true), max(y_true)],
        [min(y_true), max(y_true)],
        color="red",
        linestyle="--",
        label="Ideal",
    )
    plt.xlabel("True pIC50")
    plt.ylabel("Predicted pIC50")
    plt.title("Predicted vs. True pIC50")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def flatten_pipeline_info(pipeline):
    flat_info = {}
    for name, step in pipeline.named_steps.items():
        params = step.get_params(deep=False)
        param_str = ", ".join(
            f"{k}={v if isinstance(v, (int,float,str,bool,type(None))) else str(v)}"
            for k, v in params.items()
        )
        flat_info[f"{name}_class"] = step.__class__.__name__
        flat_info[f"{name}_params"] = param_str if param_str else "No parameters"
    return flat_info
