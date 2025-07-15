import gzip
import math
import pickle
from typing import List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

from utils.utils import get_project_path, pjoin


class SyntheticAccessibilityScorer:
    """Computes Synthetic Accessibility (SA) score for RDKit molecules."""

    def __init__(
        self,
        score_file_name: str = "fpscores.pkl.gz",
        morgan_radius: int = 2,
        score_min: float = -4.0,
        score_max: float = 2.5,
        score_lower_bound: float = 1.0,
        score_upper_bound: float = 10.0,
        score_smooth_threshold: float = 8.0,
    ):
        self.score_min = score_min
        self.score_max = score_max
        self.score_lower_bound = score_lower_bound
        self.score_upper_bound = score_upper_bound
        self.score_smooth_threshold = score_smooth_threshold

        score_file = pjoin(get_project_path(), "resources", score_file_name)
        self.fragment_scores = self._load_fragment_scores(score_file)
        self.fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=morgan_radius
        )

    def _load_fragment_scores(self, filename: str) -> dict:
        """Load precomputed fragment scores from compressed pickle file."""
        path = pjoin(get_project_path(), "resources", filename)
        with gzip.open(path, "rb") as f:
            raw_data = pickle.load(f)

        scores = {}
        for entry in raw_data:
            for fragment in entry[1:]:
                scores[fragment] = float(entry[0])
        return scores

    @staticmethod
    def count_bridgeheads_and_spiro(mol) -> Tuple[int, int]:
        """Count bridgehead and spiro atoms in the molecule."""
        n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        n_bridge = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        return n_bridge, n_spiro

    def score(self, mol) -> Optional[float]:
        """Compute the SA score (1 = easy to synthesize, 10 = hard)."""
        if not mol or mol.GetNumAtoms() == 0:
            return None

        # --- Fragment score ---
        fingerprint = self.fingerprint_generator.GetSparseCountFingerprint(mol)
        raw_fragments = fingerprint.GetNonzeroElements()
        total_score = 0.0
        total_count = 0

        for fid, count in raw_fragments.items():
            total_count += count
            total_score += self.fragment_scores.get(fid, self.score_min) * count

        score1 = total_score / max(total_count, 1)

        # --- Complexity penalties ---
        n_atoms = mol.GetNumAtoms()
        n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        ring_info = mol.GetRingInfo()
        n_bridge, n_spiro = self.count_bridgeheads_and_spiro(mol)
        n_macrocycles = sum(1 for ring in ring_info.AtomRings() if len(ring) > 8)

        size_penalty = n_atoms**1.005 - n_atoms
        stereo_penalty = math.log10(n_chiral + 1)
        spiro_penalty = math.log10(n_spiro + 1)
        bridge_penalty = math.log10(n_bridge + 1)
        macrocycle_penalty = math.log10(2) if n_macrocycles > 0 else 0.0

        score2 = -(
            size_penalty
            + stereo_penalty
            + spiro_penalty
            + bridge_penalty
            + macrocycle_penalty
        )

        # --- Symmetry correction ---
        n_bits = len(raw_fragments)
        score3 = 0.0
        if n_atoms > n_bits:
            score3 = math.log(float(n_atoms) / n_bits) * 0.5

        # --- Combine and normalize ---
        raw_sascore = score1 + score2 + score3

        sascore = (
            11.0
            - ((raw_sascore - self.score_min + 1) / (self.score_max - self.score_min))
            * 9.0
        )

        # Smooth top end
        if sascore > self.score_smooth_threshold:
            sascore = self.score_smooth_threshold + math.log(sascore + 1.0 - 9.0)

        return max(self.score_lower_bound, min(self.score_upper_bound, sascore))

    def batch_score(
        self, mols: List[Chem.Mol]
    ) -> List[Tuple[str, str, Optional[float]]]:
        """Compute SA scores for a list of molecules and return results."""
        results = []
        for mol in mols:
            if mol is None:
                continue
            score = self.score(mol)
            smiles = Chem.MolToSmiles(mol)
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else "Unknown"
            results.append((smiles, name, score))
        return results
