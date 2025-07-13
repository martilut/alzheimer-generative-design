import math
import pickle
import gzip
import os.path as op

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from utils.utils import get_project_path, pjoin

# Globals
_fscores = None
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)


def readFragmentScores(filename="fpscores.pkl.gz"):
    """Load fragment scores from a gzipped pickle file."""
    global _fscores
    with gzip.open(
            pjoin(
                get_project_path(), "resources", filename), "rb") as f:
        raw_data = pickle.load(f)

    scores = {}
    for entry in raw_data:
        for fragment in entry[1:]:
            scores[fragment] = float(entry[0])
    _fscores = scores


def numBridgeheadsAndSpiro(mol):
    """Return the number of bridgehead and spiro atoms in a molecule."""
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    n_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return n_bridgehead, n_spiro


def calculateScore(mol):
    """Calculate the synthetic accessibility score (1 = easy, 10 = hard)."""
    if not mol or not mol.GetNumAtoms():
        return None

    if _fscores is None:
        readFragmentScores()

    # Fragment-based score
    sfp = mfpgen.GetSparseCountFingerprint(mol)
    score1, nf = 0.0, 0
    for fid, count in sfp.GetNonzeroElements().items():
        nf += count
        score1 += _fscores.get(fid, -4) * count
    score1 /= max(nf, 1)

    # Complexity penalties
    n_atoms = mol.GetNumAtoms()
    n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    ring_info = mol.GetRingInfo()
    n_bridge, n_spiro = numBridgeheadsAndSpiro(mol)

    n_macrocycles = sum(1 for ring in ring_info.AtomRings() if len(ring) > 8)

    size_penalty = n_atoms**1.005 - n_atoms
    stereo_penalty = math.log10(n_chiral + 1)
    spiro_penalty = math.log10(n_spiro + 1)
    bridge_penalty = math.log10(n_bridge + 1)
    macrocycle_penalty = math.log10(2) if n_macrocycles > 0 else 0.0

    score2 = -(size_penalty + stereo_penalty + spiro_penalty + bridge_penalty + macrocycle_penalty)

    # Symmetry correction (fingerprint density)
    n_bits = len(sfp.GetNonzeroElements())
    score3 = 0.0
    if n_atoms > n_bits:
        score3 = math.log(float(n_atoms) / n_bits) * 0.5

    raw_sascore = score1 + score2 + score3

    # Rescale to 1–10
    min_score, max_score = -4.0, 2.5
    sascore = 11.0 - ((raw_sascore - min_score + 1) / (max_score - min_score)) * 9.0

    # Smooth upper bound
    if sascore > 8.0:
        sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
    sascore = max(1.0, min(10.0, sascore))

    return sascore


def processMols(mols):
    """Calculate SA score for each molecule and print."""
    print("SMILES\tName\tSA_Score")
    for mol in mols:
        if mol is None:
            continue

        score = calculateScore(mol)
        smiles = Chem.MolToSmiles(mol)
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else "Unknown"

        if score is None:
            print(f"{smiles}\t{name}\tNone")
        else:
            print(f"{smiles}\t{name}\t{score:.3f}")
