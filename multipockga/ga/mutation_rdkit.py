from typing import Optional
import random

import selfies as sf
from rdkit import Chem


ALPHABET = list(sf.get_semantic_robust_alphabet())


def canonicalize(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def mutate_selfies(
    smiles: str,
    n_tries: int = 30,
    rng: Optional[random.Random] = None,
) -> Optional[str]:
    rng = rng or random
    try:
        selfie = sf.encoder(smiles)
        tokens = list(sf.split_selfies(selfie))
    except Exception:
        return None

    if not tokens:
        return None

    for _ in range(n_tries):
        toks = tokens[:]
        op = rng.choice(["replace", "insert", "delete"])

        if op == "replace" and toks:
            toks[rng.randrange(len(toks))] = rng.choice(ALPHABET)
        elif op == "insert":
            toks.insert(rng.randrange(len(toks) + 1), rng.choice(ALPHABET))
        elif op == "delete" and len(toks) > 1:
            toks.pop(rng.randrange(len(toks)))
        else:
            continue

        try:
            mutated = sf.decoder("".join(toks))
        except Exception:
            continue

        mutated = canonicalize(mutated)
        if mutated and mutated != smiles:
            return mutated

    return None


def generate_mutations(seed_smiles, budget: int, seed: Optional[int] = None):
    rng = random.Random(seed)
    if not seed_smiles or budget <= 0:
        return set()

    generated = set()
    for _ in range(budget):
        parent = rng.choice(seed_smiles)
        child = mutate_selfies(parent, rng=rng)
        if child:
            generated.add(child)
    return generated