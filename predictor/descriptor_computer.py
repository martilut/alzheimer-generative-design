import traceback

import pandas as pd
from mordred import Calculator, descriptors
from rdkit.Chem import Descriptors

from utils.utils import get_data_folder, pjoin, get_molecule


def get_rdkit_descriptors(molecule) -> dict:
    res = {}
    for name, func in Descriptors._descList:
        try:
            value = func(molecule)
        except Exception as e:
            traceback.print_exc()
            value = None
        res[name] = value
    return res


def get_mordred_descriptors(molecule) -> dict:
    calc = Calculator(descriptors)
    res = {}
    for desc, value in calc(molecule).items():
        try:
            res[str(desc)] = value
        except Exception as e:
            traceback.print_exc()
            res[str(desc)] = None
    return res


DESC_FUNCS = {
    "rdkit": get_rdkit_descriptors,
    "mordred": get_mordred_descriptors,
}


def get_desc_dataset(target_id, desc_func_name) -> pd.DataFrame:
    path = pjoin(get_data_folder(), f"{target_id}_{desc_func_name}_desc.csv")
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
        desc = DESC_FUNCS[desc_func_name](mol)
        res.update({f"{k}": v for k, v in desc.items() if v is not None})
        dataset.append(res)
    dataset = pd.DataFrame(dataset)
    dataset.set_index("molecule_chembl_id", drop=True, inplace=True)

    dataset.to_csv(path, index=True)
    return dataset


if __name__ == "__main__":
    print(get_desc_dataset("CHEMBL2760", "mordred"))
