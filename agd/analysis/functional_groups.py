import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

def count_functional_groups(file_path):
    """
    Counts various functional groups and features in molecules from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing a 'smiles' column.

    Returns:
        dict: A dictionary with counts of molecules containing each functional group or feature.
    """

    # Load the CSV file
    df = pd.read_csv(file_path)

    # Initialize counters for each functional group or feature
    counts = {
        "amines_primary": 0,
        "amines_secondary": 0,
        "amines_tertiary": 0,
        "amides": 0,
        "carboxylic_acids": 0,
        "esters": 0,
        "nitriles": 0,
        "nitro_groups": 0,
        "sulfonamides": 0,
        "hydroxyl_groups": 0,
        "phenols": 0,
        "thiols": 0,
        "heterocycles": 0,
        "fluorine_atoms": 0,
        "chlorine_atoms": 0,
        "bromine_atoms": 0,
        "alkynes": 0,
        "aromatic_rings": 0,
    }

    # SMARTS patterns for searching functional groups
    smarts_patterns = {
        "amines_primary": "[NX3;H2]",                 # primary amine
        "amines_secondary": "[NX3;H1]",               # secondary amine
        "amines_tertiary": "[NX3;H0;!$(NC=O)]",       # tertiary amine (not an amide)
        "amides": "NC=O",
        "carboxylic_acids": "C(=O)[OH]",
        "esters": "C(=O)O[C;!H0]",
        "nitriles": "C#N",
        "nitro_groups": "[$([NX3](=O)=O)]",
        "sulfonamides": "S(=O)(=O)N",
        "hydroxyl_groups": "[OX2H]",                  # alcohol OH groups
        "phenols": "c[OH]",                           # phenols
        "thiols": "[SX2H]",                           # SH groups
        "alkynes": "C#C",
    }

    # Iterate over each SMILES string in the dataframe
    for smiles in df["smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        # Check presence of each functional group using SMARTS matching
        for name, smarts in smarts_patterns.items():
            patt = Chem.MolFromSmarts(smarts)
            if mol.HasSubstructMatch(patt):
                counts[name] += 1

        # Count heterocyclic rings
        num_heterocycles = rdMolDescriptors.CalcNumHeterocycles(mol)
        if num_heterocycles > 0:
            counts["heterocycles"] += 1

        # Check for presence of fluorine, chlorine, bromine atoms
        if any(atom.GetSymbol() == "F" for atom in mol.GetAtoms()):
            counts["fluorine_atoms"] += 1

        if any(atom.GetSymbol() == "Cl" for atom in mol.GetAtoms()):
            counts["chlorine_atoms"] += 1

        if any(atom.GetSymbol() == "Br" for atom in mol.GetAtoms()):
            counts["bromine_atoms"] += 1

        # Count aromatic rings
        num_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
        if num_aromatic > 0:
            counts["aromatic_rings"] += 1

    return counts
