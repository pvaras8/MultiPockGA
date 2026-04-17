from typing import Optional
import random

from rdkit import Chem
from rdkit.Chem import BRICS


def canonicalize(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def brics_fragments(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    try:
        return list(BRICS.BRICSDecompose(mol))
    except Exception:
        return []


def crossover_pair(smiles_a: str, smiles_b: str, max_products: int = 8):
    frags_a = brics_fragments(smiles_a)
    frags_b = brics_fragments(smiles_b)
    if not frags_a or not frags_b:
        return []

    pool = list(set(frags_a + frags_b))
    mols = [Chem.MolFromSmiles(f) for f in pool]
    mols = [m for m in mols if m is not None]
    if len(mols) < 2:
        return []

    out = []
    try:
        builder = BRICS.BRICSBuild(mols)
        for mol in builder:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
            smi = canonicalize(smi)
            if smi:
                out.append(smi)
            if len(out) >= max_products:
                break
    except Exception:
        return []

    return list(dict.fromkeys(out))


def generate_crossovers(
    parent_smiles,
    budget: int,
    max_products_per_pair: int = 4,
    seed: Optional[int] = None,
):
    rng = random.Random(seed)
    if len(parent_smiles) < 2 or budget <= 0:
        return set()

    generated = set()
    for _ in range(budget):
        p1, p2 = rng.sample(parent_smiles, 2)
        for child in crossover_pair(p1, p2, max_products=max_products_per_pair):
            if child != p1 and child != p2:
                generated.add(child)
    return generated