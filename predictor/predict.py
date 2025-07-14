import pandas as pd
from rdkit import Chem

from predictor.descriptor_computer import DESC_FUNCS
from utils.utils import pjoin, get_project_path
import joblib


def predict(smiles: str, model_name: str, desc_func_name: str) -> float:
    load_path = pjoin(get_project_path(), "resources", f"{model_name}.pkl")
    pipeline = joblib.load(load_path)

    mol = Chem.MolFromSmiles(smiles)
    descriptors = DESC_FUNCS[desc_func_name](mol)

    X = pd.DataFrame(descriptors, index=[smiles])
    pipeline_features = list(pipeline.named_steps['preprocessing'].get_feature_names_out())
    X = X.loc[:, pipeline_features]

    return pipeline.predict(X)[0]
