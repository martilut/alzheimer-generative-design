import pandas as pd
from rdkit import Chem

from predictor.descriptor_computer import get_rdkit_descriptors
from utils.utils import pjoin, get_project_path
import joblib


def predict(smiles: str, model_name: str) -> float:
    load_path = pjoin(get_project_path(), "resources", f"{model_name}.pkl")
    pipeline = joblib.load(load_path)

    mol = Chem.MolFromSmiles(smiles)
    descriptors = get_rdkit_descriptors(mol)

    X = pd.DataFrame(descriptors, index=[smiles])
    X = X.loc[:, pipeline.named_steps['preprocessing'].get_feature_names_out()]

    return pipeline.predict(X)[0]
