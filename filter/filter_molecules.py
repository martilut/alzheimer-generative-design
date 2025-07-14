import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, AllChem, Lipinski
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from filter.sascorer import calculateScore


### Рассчет параметров


# Считаем QED
def QED_definer(smiles):
    mol = Chem.MolFromSmiles(smiles)
    qed_value = QED.qed(mol)
    return qed_value


# Считаем SA_SCORER
def sa_scorer_definer(mol):
    return calculateScore(mol)


# Фиксируем нарушения правил Липинского
def lipinski_definer(mol):
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)

    violations = 0
    if mw > 500:
        violations += 1
    if logp > 5:
        violations += 1
    if h_donors > 5:
        violations += 1
    if h_acceptors > 10:
        violations += 1

    return violations


# Ищем токсикофоры и рассчитываем BBB
def compute_properties(mol):

    if not mol:
        return {"tox_free": False, "bbb": False}

    # --- Токсикофоры ---
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.NIH)
    catalog = FilterCatalog(params)
    toxicophore_free = not catalog.HasMatch(mol)

    # --- BBB критерии ---
    mol_weight = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    bbb_pass = (400 <= mol_weight <= 500) and (logp > 1)

    return {"tox_free": toxicophore_free, "bbb": bbb_pass}


# Проверка на канцерогенность
def carcinogenicity_check(mol):
    if not mol:
        return False

    # Определяем SMARTS-паттерны канцерогенных групп
    smarts_patterns = [
        "[NX3][C](=[O])[NX3]",  # нитрозамины
        "c1ccc(N)cc1",  # ароматические амины
        "[NX3]=[NX3]",  # азо-соединения
        "[O;D2]-[N+](=O)[O-]",  # нитрогруппы
    ]

    for smarts in smarts_patterns:
        patt = Chem.MolFromSmarts(smarts)
        if patt and mol.HasSubstructMatch(patt):
            return True
    return False


def filter_molecules(df: pd.DataFrame):

    # --- Apply computations ---

    df_filtered = []
    for mol_idx in df.index:
        smiles = df.loc[mol_idx, "canonical_smiles"]
        ic50 = df.loc[mol_idx, "standard_value"]
        res = {
            "molecule_chembl_id": mol_idx,
            "smiles": smiles,
            "ic50": ic50,
        }
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES for {mol_idx}: {smiles}")
            continue
        props = compute_properties(mol)
        res.update(props)

        res["qed"] = QED_definer(smiles)
        res["sa"] = sa_scorer_definer(mol)
        res["lip"] = lipinski_definer(mol)
        res["carc"] = not carcinogenicity_check(mol)

        df_filtered.append(res)
    df_filtered = pd.DataFrame(df_filtered)

    # --- Filter conditions ---

    df_filtered = df_filtered[
        (df_filtered["qed"] > 0.7)
        & (df_filtered["sa"].between(2, 6))
        & (df_filtered["bbb"])
        & (df_filtered["tox_free"])
        & (df_filtered["lip"] <= 1)
        & (df_filtered["carc"])
    ]

    return df_filtered
