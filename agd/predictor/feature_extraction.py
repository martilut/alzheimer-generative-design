import os
import traceback

import numpy as np
import pandas as pd
from mordred import Calculator
from mordred import descriptors as mordred_descriptors
from rdkit import DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from agd.utils.utils import get_data_folder, get_molecule, pjoin


def get_fingerprint_array(mol, fp_generator) -> np.ndarray:
    fp = fp_generator.GetFingerprint(mol)
    fp_array = np.zeros((fp.GetNumBits(),), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, fp_array)
    return fp_array


def get_rdkit_descriptors(mol) -> dict:
    res = {}
    for name, func in Descriptors._descList:
        try:
            res[name] = func(mol)
        except Exception:
            traceback.print_exc()
            res[name] = None
    return res


def get_mordred_descriptors(mol) -> dict:
    calc = Calculator(mordred_descriptors)
    res = {}
    try:
        desc_values = calc(mol)
        for desc, value in desc_values.items():
            try:
                res[str(desc)] = value
            except Exception:
                traceback.print_exc()
                res[str(desc)] = None
    except Exception:
        traceback.print_exc()
    return res


DESC_FUNCS = {
    "rdkit": get_rdkit_descriptors,
    "mordred": get_mordred_descriptors,
}

FP_GENERATORS = {
    "morgan": rdFingerprintGenerator.GetMorganGenerator,
    # Add more if needed
}


def describe_fp(name: str, params: dict) -> str:
    parts = [name]
    for k, v in sorted(params.items()):
        parts.append(str(v))
    return "_".join(parts)


def smiles_to_features(
    smiles_list,
    descriptors: list = None,
    fingerprints: dict = None,
) -> pd.DataFrame:
    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]

    all_feature_dfs = []

    if descriptors:
        for desc_name in descriptors:
            if desc_name not in DESC_FUNCS:
                raise ValueError(f"Unsupported descriptor: {desc_name}")
            desc_func = DESC_FUNCS[desc_name]

            desc_records = []
            for smiles in smiles_list:
                mol = get_molecule(smiles)
                if mol is None:
                    print(f"Invalid SMILES: {smiles}")
                    desc_records.append({})
                    continue
                desc = desc_func(mol)
                desc = {f"{k}": v for k, v in desc.items() if v is not None}
                desc_records.append(desc)

            all_feature_dfs.append(pd.DataFrame(desc_records))

    if fingerprints:
        for fp_name, fp_info in fingerprints.items():
            if "params" not in fp_info:
                raise ValueError(f"Missing 'params' for fingerprint: {fp_name}")
            if fp_name not in FP_GENERATORS:
                raise ValueError(f"Unknown fingerprint generator: {fp_name}")

            fp_gen_class = FP_GENERATORS[fp_name]
            try:
                fp_gen = fp_gen_class(**fp_info["params"])
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize fingerprint generator '{fp_name}': {e}"
                )

            fp_records = []
            for smiles in smiles_list:
                mol = get_molecule(smiles)
                if mol is None:
                    print(f"Invalid SMILES: {smiles}")
                    fp_records.append({})
                    continue
                fp_array = get_fingerprint_array(mol, fp_gen)
                fp_dict = {f"{i}": v for i, v in enumerate(fp_array)}
                fp_records.append(fp_dict)

            all_feature_dfs.append(pd.DataFrame(fp_records))

    if not all_feature_dfs:
        raise ValueError(
            "No features computed. Provide at least descriptors or fingerprints."
        )

    combined_df = pd.concat(all_feature_dfs, axis=1)
    combined_df.reset_index(drop=True, inplace=True)
    return combined_df


def get_features_dataset(
    target_id: str,
    descriptors: list = None,
    fingerprints: dict = None,
    rewrite: bool = False,
    data_dir: str = None,
) -> pd.DataFrame:
    feature_suffix_parts = []

    if descriptors:
        feature_suffix_parts += descriptors
    if fingerprints:
        for fp_name, fp_info in fingerprints.items():
            fp_desc = describe_fp(fp_name, fp_info.get("params", {}))
            feature_suffix_parts.append(fp_desc)

    feature_suffix = "_".join(feature_suffix_parts)
    if data_dir is None:
        path = pjoin(get_data_folder(), f"{target_id}_{feature_suffix}.csv")
        df = pd.read_csv(
            pjoin(get_data_folder(), f"{target_id}_clean.csv"), index_col=0
        )
    else:
        path = pjoin(data_dir, f"{target_id}_{feature_suffix}.csv")
        df = pd.read_csv(pjoin(data_dir, f"{target_id}_clean.csv"), index_col=0)

    if os.path.exists(path) and not rewrite:
        return pd.read_csv(path, index_col=0)

    smiles_list = df["canonical_smiles"].tolist()

    features_df = smiles_to_features(smiles_list, descriptors, fingerprints)

    features_df.index = df.index
    features_df.insert(0, "smiles", df["canonical_smiles"])
    features_df.insert(1, "standard_value", df["standard_value"])

    features_df.to_csv(path)
    return features_df
