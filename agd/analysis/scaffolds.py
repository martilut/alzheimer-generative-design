import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def add_scaffolds_to_csv(input_csv, output_csv):
    """
    Reads a CSV with SMILES strings, generates Murcko scaffolds for each molecule,
    adds them as a new column, and saves the updated CSV.

    Args:
        input_csv (str): Path to the input CSV file with a 'smiles' column.
        output_csv (str): Path to save the new CSV with scaffold column added.

    Returns:
        set: A set of unique scaffold SMILES strings found in the dataset.
    """
    # Load the CSV file
    df = pd.read_csv(input_csv)

    # List to store scaffold SMILES
    scaffolds = []

    # Iterate over each SMILES string in the dataframe
    for smiles in df["smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold)
            scaffolds.append(scaffold_smiles)
        else:
            scaffolds.append(None)  # In case the SMILES string is invalid

    # Add the scaffold column to the dataframe
    df["scaffold"] = scaffolds

    # Save the updated dataframe to a new CSV file
    df.to_csv(output_csv, index=False)

    # Get unique scaffolds (excluding None)
    unique_scaffolds = set(s for s in scaffolds if s is not None)

    return unique_scaffolds
