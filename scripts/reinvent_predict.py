import json
import pickle
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors

# Load your trained pipeline
with open("../resources/catboost_rdkit_morgan.pkl", "rb") as f:
    model = pickle.load(f)

# Read SMILES from stdin
smiles_list = [line.strip() for line in sys.stdin if line.strip()]

# Get all RDKit 2D descriptor functions
descriptor_names = [desc[0] for desc in Descriptors._descList]
descriptor_funcs = [desc[1] for desc in Descriptors._descList]


def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Calculate all RDKit descriptors
    desc_values = []
    for func in descriptor_funcs:
        try:
            val = func(mol)
        except Exception:
            val = np.nan
        desc_values.append(val)

    # Morgan fingerprint radius=3, 1024 bits
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024)
    fp_arr = np.zeros((1024,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)

    # Combine descriptors and fingerprint into single vector
    features = np.concatenate([desc_values, fp_arr])

    return features


features = []
for smi in smiles_list:
    feats = featurize(smi)
    if feats is None:
        # NaNs for descriptors + zeros for fingerprint
        nan_part = np.full(len(descriptor_funcs), np.nan)
        zero_fp = np.zeros(1024)
        features.append(np.concatenate([nan_part, zero_fp]))
    else:
        features.append(feats)

X = np.array(features)

# Predict
predictions = model.predict(X).tolist()

# Output results in required JSON format
result = {"version": 1, "payload": {"predictions": predictions}}

print(json.dumps(result))
