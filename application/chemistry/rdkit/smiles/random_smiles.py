# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: random_smiles.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/9/25 16:27

from typing import List
import random
from rdkit import Chem


def _augment_smiles(smiles: str, num_variants=1) -> List[str]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [smiles] * num_variants
        results = []
        for _ in range(num_variants):
            random_smiles = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
            results.append(random_smiles)
        return results
    except:
        return [smiles] * num_variants

if __name__ == '__main__':
    smiles = "N[C@H]1CCC[C@@H]1Nc1cnc(C(F)(F)F)cn1.O=C(O)c1cc(Cl)ccc1-n1cccn1"
    augmented_smiles = _augment_smiles(smiles, num_variants=5)
    print(augmented_smiles)