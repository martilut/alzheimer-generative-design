import pytest
import pandas as pd
from rdkit import Chem

from agd.filter.molecule_evaluator import MoleculeEvaluator, filter_molecules


# Example SMILES for testing
VALID_SMILES = "N#CCC1(n2cc(C(N)=O)c(=Nc3ccnc(C(F)(F)F)c3)[nH]2)CCC(N)CC1"
INVALID_SMILES = "not_a_smiles"
CARC_SMILES = "c1ccc(N)cc1"
LIPINSKI_FAIL_SMILES = "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"

@pytest.fixture
def valid_mol():
    return Chem.MolFromSmiles(VALID_SMILES)

@pytest.fixture
def carcinogenic_mol():
    return Chem.MolFromSmiles(CARC_SMILES)

@pytest.fixture
def evaluator():
    return MoleculeEvaluator()


def test_compute_qed(evaluator):
    qed = evaluator.compute_qed(VALID_SMILES)
    assert 0.0 < qed <= 1.0

def test_compute_sa_score(evaluator, valid_mol):
    sa = evaluator.compute_sa_score(valid_mol)
    assert 1.0 <= sa <= 10.0

def test_count_lipinski_violations_pass(evaluator, valid_mol):
    violations = evaluator.count_lipinski_violations(valid_mol)
    assert violations <= evaluator.lipinski_max_violations

def test_assess_properties_keys(evaluator, valid_mol):
    props = evaluator.assess_properties(valid_mol)
    assert "tox_free" in props and "bbb" in props
    assert isinstance(props["tox_free"], bool)
    assert isinstance(props["bbb"], bool)

def test_is_carcinogenic_positive(evaluator, carcinogenic_mol):
    assert evaluator.is_carcinogenic(carcinogenic_mol) is True

def test_is_carcinogenic_negative(evaluator, valid_mol):
    assert evaluator.is_carcinogenic(valid_mol) is False

def test_filter_molecules_filters_properly(evaluator):
    df = pd.DataFrame([
        {"canonical_smiles": VALID_SMILES, "standard_value": 0.05},
        {"canonical_smiles": INVALID_SMILES, "standard_value": 0.1},
        {"canonical_smiles": CARC_SMILES, "standard_value": 0.5},
        {"canonical_smiles": LIPINSKI_FAIL_SMILES, "standard_value": 0.2},
    ])
    filtered_df = filter_molecules(df, evaluator)

    assert isinstance(filtered_df, pd.DataFrame)
    assert all(col in filtered_df.columns for col in [
        "molecule_chembl_id", "smiles", "ic50", "qed", "sa",
        "lip", "tox_free", "bbb", "carc"
    ])
    assert VALID_SMILES in filtered_df["smiles"].values
    assert INVALID_SMILES not in filtered_df["smiles"].values
    assert CARC_SMILES not in filtered_df["smiles"].values
    assert LIPINSKI_FAIL_SMILES not in filtered_df["smiles"].values
