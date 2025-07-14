import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
import traceback
import os
from mordred import Calculator, descriptors

from utils.utils import get_project_path, get_data_folder, pjoin, get_molecule


def get_fp(mol, fp_generator):
    try:
        fp = fp_generator.GetFingerprint(mol)
    except Exception as e:
        fp = None
    return fp

def get_fp_dataset(target_id, fp_name, fp_generator) -> pd.DataFrame:
    path = pjoin(get_data_folder(), f"{target_id}_{fp_name}_fp.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    df = pd.read_csv(pjoin(get_data_folder(), f"{target_id}_clean.csv"), index_col=0)
    dataset = []

    for molecule_id in df.index:
        molecule_smiles = df.loc[molecule_id, "canonical_smiles"]
        molecule_ic50 = df.loc[molecule_id, "standard_value"]
        res = {
            "molecule_chembl_id": molecule_id,
            "smiles": molecule_smiles,
            "standard_value": molecule_ic50,
        }
        mol = get_molecule(molecule_smiles)
        if mol is None:
            print(f"Invalid SMILES for {molecule_id}: {molecule_smiles}")
            continue

        fp = get_fp(mol, fp_generator)
        if fp is None:
            print(f"Failed to compute fingerprint for {molecule_id}: {molecule_smiles}")
            continue
        fp_array = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, fp_array)
        res.update(
            {f"{i}": v for i, v in enumerate(fp_array)}
        )
        dataset.append(res)
    dataset = pd.DataFrame(dataset)
    dataset.set_index("molecule_chembl_id", drop=True, inplace=True)
    dataset.to_csv(path, index=True)
    return dataset


if __name__ == "__main__":
    target_id = "CHEMBL2760"
    fp_name = "morgan"
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)

    # Generate the fingerprint dataset
    fp_dataset = get_fp_dataset(target_id, fp_name, fp_generator)
    print(fp_dataset.head())
