import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from agd.filter.sa_scorer import SyntheticAccessibilityScorer


class MoleculeEvaluator:
    """Encapsulates molecular property evaluations and filters."""

    def __init__(
        self,
        qed_threshold: float = 0.7,
        sa_score_range: tuple = (2.0, 6.0),
        bbb_mw_range: tuple = (400, 500),
        bbb_logp_min: float = 1.0,
        lipinski_mw_max: float = 500,
        lipinski_logp_max: float = 5,
        lipinski_h_donors_max: int = 5,
        lipinski_h_acceptors_max: int = 10,
        lipinski_max_violations: int = 1,
        sa_scorer: SyntheticAccessibilityScorer | None = None
    ):
        self.qed_threshold = qed_threshold
        self.sa_score_range = sa_score_range
        self.bbb_mw_range = bbb_mw_range
        self.bbb_logp_min = bbb_logp_min
        self.lipinski_mw_max = lipinski_mw_max
        self.lipinski_logp_max = lipinski_logp_max
        self.lipinski_h_donors_max = lipinski_h_donors_max
        self.lipinski_h_acceptors_max = lipinski_h_acceptors_max
        self.lipinski_max_violations = lipinski_max_violations
        self.sa_scorer = SyntheticAccessibilityScorer() if sa_scorer is None else sa_scorer
        self.carcinogenic_patterns = [
            "[NX3][C](=[O])[NX3]",  # Nitrosamines
            "c1ccc(N)cc1",  # Aromatic amines
            "[NX3]=[NX3]",  # Azo compounds
            "[O;D2]-[N+](=O)[O-]"  # Nitro groups
        ]

    def compute_sa_score(self, mol) -> float:
        """Compute Synthetic Accessibility (SA) score."""
        return self.sa_scorer.score(mol)

    @staticmethod
    def compute_qed(smiles: str) -> float:
        mol = Chem.MolFromSmiles(smiles)
        return QED.qed(mol) if mol else 0.0

    def count_lipinski_violations(self, mol) -> int:
        violations = 0
        if Descriptors.MolWt(mol) > self.lipinski_mw_max:
            violations += 1
        if Descriptors.MolLogP(mol) > self.lipinski_logp_max:
            violations += 1
        if Descriptors.NumHDonors(mol) > self.lipinski_h_donors_max:
            violations += 1
        if Descriptors.NumHAcceptors(mol) > self.lipinski_h_acceptors_max:
            violations += 1
        return violations

    def assess_properties(self, mol) -> dict:
        if not mol:
            return {"tox_free": False, "bbb": False}

        # Toxicophore filtering
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.NIH)
        catalog = FilterCatalog(params)
        tox_free = not catalog.HasMatch(mol)

        # BBB check
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        bbb = (self.bbb_mw_range[0] <= mw <= self.bbb_mw_range[1]) and (logp > self.bbb_logp_min)

        return {"tox_free": tox_free, "bbb": bbb}


    def is_carcinogenic(self, mol) -> bool:
        if not mol:
            return False
        for smarts in self.carcinogenic_patterns:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                return True
        return False


def filter_molecules(df: pd.DataFrame, evaluator: MoleculeEvaluator) -> pd.DataFrame:
    """
    Filter molecules based on drug-likeness, safety, and ADME criteria.
    """

    filtered_data = []

    for idx, row in df.iterrows():
        smiles = row.get("canonical_smiles")
        ic50 = row.get("standard_value")
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            print(f"Invalid SMILES at {idx}: {smiles}")
            continue

        props = evaluator.assess_properties(mol)

        molecule_record = {
            "molecule_chembl_id": idx,
            "smiles": smiles,
            "ic50": ic50,
            "qed": evaluator.compute_qed(smiles),
            "sa": evaluator.compute_sa_score(mol),
            "lip": evaluator.count_lipinski_violations(mol),
            "tox_free": props["tox_free"],
            "bbb": props["bbb"],
            "carc": not evaluator.is_carcinogenic(mol),
        }

        filtered_data.append(molecule_record)

    result_df = pd.DataFrame(filtered_data)

    return result_df[
        (result_df["qed"] > evaluator.qed_threshold) &
        (result_df["sa"].between(*evaluator.sa_score_range)) &
        (result_df["bbb"]) &
        (result_df["tox_free"]) &
        (result_df["lip"] <= evaluator.lipinski_max_violations) &
        (result_df["carc"])
    ]
