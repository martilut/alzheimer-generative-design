import joblib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from predictor.descriptor_computer import DESC_FUNCS, get_rdkit_descriptors
from predictor.fingerprint_computer import get_fp, get_fp_dataset
from utils.utils import get_project_path, pjoin


def predict(smiles: str, model_name: str, desc_func_names: list[str]) -> float:
    load_path = pjoin(get_project_path(), "resources", f"{model_name}.pkl")
    pipeline = joblib.load(load_path)

    mol = Chem.MolFromSmiles(smiles)
    descriptors = []
    for desc_func_name in desc_func_names:
        if desc_func_name not in DESC_FUNCS:
            raise ValueError(f"Descriptor function '{desc_func_name}' not found.")
        desc_func = DESC_FUNCS[desc_func_name]
        desc_value = desc_func(mol)
        desc_value = pd.DataFrame(desc_value, index=[smiles])
        descriptors.append(desc_value)
    descriptors = pd.concat(descriptors, axis=1)

    pipeline_features = list(
        pipeline.named_steps["preprocessing"].get_feature_names_out()
    )
    x = descriptors.loc[:, pipeline_features]

    return pipeline.predict(x)[0]


def predict_rdkit_morgan(
    smiles: str, model_name: str, bit: int = 1024, radius: int = 3
) -> float:
    load_path = pjoin(get_project_path(), "resources", f"{model_name}_rdkit_morgan.pkl")
    pipeline = joblib.load(load_path)

    mol = Chem.MolFromSmiles(smiles)
    rdkit_desc = pd.DataFrame(get_rdkit_descriptors(mol), index=[smiles])
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bit)
    fp = get_fp(mol, fp_gen)
    morgan_fp = pd.DataFrame(fp.reshape((1, bit)), index=[smiles])
    descriptors = pd.concat([rdkit_desc, morgan_fp], axis=1)

    pipeline_features = list(pipeline.named_steps["model"].feature_names_)
    x = descriptors.copy()
    x.columns = [str(i) for i in x.columns]
    x = x.loc[:, pipeline_features]

    return pipeline.predict(x)[0]


if __name__ == "__main__":
    print(
        predict_rdkit_morgan(
            smiles="CN[C@@H]1C[C@H]2O[C@@](C)([C@@H]1OC)n1c3ccccc3c3c4c(c5c6ccccc6n2c5c31)C(=O)NC4",
            model_name="catboost",
        )
    )
