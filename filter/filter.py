import requests
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import QED
from rdkit.Chem import Descriptors
from rdkit.Chem import QED, Lipinski, Descriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
import sascorer


def filter(df_hits):
    ### Рассчет параметров

    # Считаем QED
    def QED_definer(smiles):
        mol = Chem.MolFromSmiles(smiles)
        qed_value = QED.qed(mol)
        return qed_value

    # Считаем SA_SCORER
    def sa_scorer_definer(mol):
        return sascorer.calculateScore(mol)

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
            return {"ToxicophoreFree": False, "BBB": False}

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

        return {"ToxicophoreFree": toxicophore_free, "BBB": bbb_pass}

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

    ### Добавляем рассчеты в датафрейм

    df_props = df_hits["RDkitMol"].apply(compute_properties).apply(pd.Series)
    df_hits = pd.concat([df_hits, df_props], axis=1)
    df_hits["QED"] = df_hits["SMILES"].apply(QED_definer)
    df_hits["SA_Score"] = df_hits["RDkitMol"].apply(sa_scorer_definer)
    df_hits["Lipinski_violations"] = df_hits["RDkitMol"].apply(lipinski_definer)
    df_hits["CarcinogenicityFree"] = df_hits["RDkitMol"].apply(carcinogenicity_check)

    ### Непосредственно фильтрация
    df_hits_filtered = (
        df_hits[
            (df_hits["QED"] > 0.7)
            & (df_hits["SA_Score"].between(2, 6))
            & (df_hits["BBB"] == True)
            & (df_hits["ToxicophoreFree"] == True)
            & (df_hits["Lipinski_violations"] <= 1)
            & (df_hits["CarcinogenicityFree"] == True)
        ]
        .drop(columns={"RDkitMol"})
        .copy()
    )

    return df_hits_filtered
