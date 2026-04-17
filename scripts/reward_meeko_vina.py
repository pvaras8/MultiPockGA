import json
import os
import sys
from multiprocessing import cpu_count, get_context
from typing import List, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy
from vina import Vina


# ============================================================
# Utilidades
# ============================================================

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = json.load(f)

    required = [
        "filename_of_receptor",
        "center_x",
        "center_y",
        "center_z",
        "size_x",
        "size_y",
        "size_z",
        "final_folder",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Faltan claves en vars.json: {missing}")

    return cfg


def read_smiles_file(smiles_path: str) -> List[str]:
    smiles = []
    with open(smiles_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            smi = line.split()[0]
            smiles.append(smi)
    return smiles


def smiles_to_3d_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("SMILES inválido")

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise RuntimeError("No se pudo generar conformero 3D")

    try:
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
        if mmff_props is not None:
            AllChem.MMFFOptimizeMolecule(mol)
        else:
            AllChem.UFFOptimizeMolecule(mol)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            pass

    return mol


def mol_to_pdbqt_string(mol) -> str:
    preparator = MoleculePreparation()
    setups = preparator.prepare(mol)

    if not setups or len(setups) == 0:
        raise RuntimeError("Meeko no devolvió MoleculeSetup")

    pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setups[0])

    if not is_ok:
        raise RuntimeError(f"Error al escribir PDBQT con Meeko: {error_msg}")

    return pdbqt_string


# ============================================================
# Estado global por worker
# ============================================================

WORKER_STATE = {}


def init_worker(cfg: dict):
    global WORKER_STATE

    v = Vina(sf_name="vina", cpu=cfg["vina_cpu_per_job"])
    v.set_receptor(cfg["filename_of_receptor"])
    v.compute_vina_maps(center=cfg["center"], box_size=cfg["box_size"])

    WORKER_STATE = {
        "vina": v,
        "exhaustiveness": cfg["exhaustiveness"],
        "n_poses": cfg["n_poses"],
        "fallback_score": cfg["fallback_score"],
    }


def worker(task: Tuple[int, str]):
    idx, smiles = task

    try:
        mol = smiles_to_3d_mol(smiles)
        ligand_pdbqt = mol_to_pdbqt_string(mol)

        v = WORKER_STATE["vina"]
        v.set_ligand_from_string(ligand_pdbqt)
        v.dock(
            exhaustiveness=WORKER_STATE["exhaustiveness"],
            n_poses=WORKER_STATE["n_poses"],
        )

        energies = v.energies()
        if energies is None or len(energies) == 0:
            raise RuntimeError("Vina no devolvió energías")

        score = float(energies[0][0])

        return idx, {
            "ID": str(idx + 1),
            "SMILES": smiles,
            "Docking": score,
        }

    except Exception:
        return idx, {
            "ID": str(idx + 1),
            "SMILES": smiles,
            "Docking": WORKER_STATE["fallback_score"],
        }


# ============================================================
# Main
# ============================================================

def main():
    if len(sys.argv) != 4:
        raise SystemExit(
            "Uso: python reward_vina.py <source_compound_file> <vars.json> <epoch>"
        )

    source_compound_file = os.path.abspath(sys.argv[1])
    vars_json = os.path.abspath(sys.argv[2])
    epoch = str(sys.argv[3])

    cfg = load_config(vars_json)

    if not os.path.exists(source_compound_file):
        raise FileNotFoundError(f"No se encontró el archivo de SMILES: {source_compound_file}")

    receptor_pdbqt = os.path.abspath(cfg["filename_of_receptor"])
    if not os.path.exists(receptor_pdbqt):
        raise FileNotFoundError(f"No se encontró el receptor PDBQT: {receptor_pdbqt}")

    smiles_list = read_smiles_file(source_compound_file)
    if len(smiles_list) == 0:
        raise ValueError("El archivo de entrada no contiene SMILES")

    center = [
        float(cfg["center_x"]),
        float(cfg["center_y"]),
        float(cfg["center_z"]),
    ]
    box_size = [
        float(cfg["size_x"]),
        float(cfg["size_y"]),
        float(cfg["size_z"]),
    ]

    n_workers = int(cfg.get("num_processors", max(1, cpu_count() - 1)))
    if n_workers == -1:
        n_workers = max(1, cpu_count() - 1)
    n_workers = max(1, n_workers)

    vina_cpu_per_job = int(cfg.get("vina_cpu_per_job", 1))
    vina_cpu_per_job = max(1, vina_cpu_per_job)

    final_folder = os.path.abspath(cfg["final_folder"])
    os.makedirs(final_folder, exist_ok=True)

    output_csv_path = os.path.join(
        final_folder, f"docking_results_{epoch}_temp.csv"
    )

    worker_cfg = {
        "filename_of_receptor": receptor_pdbqt,
        "center": center,
        "box_size": box_size,
        "exhaustiveness": int(cfg.get("exhaustiveness", 8)),
        "n_poses": int(cfg.get("n_poses", 1)),
        "vina_cpu_per_job": vina_cpu_per_job,
        "fallback_score": float(cfg.get("fallback_score", -6.0)),
    }

    tasks = list(enumerate(smiles_list))

    print("---------- 1. Prediccion Docking ----------")
    print(f"Moléculas: {len(smiles_list)}")
    print(f"Workers: {n_workers}")
    print(f"Vina CPU/job: {vina_cpu_per_job}")
    print(f"Receptor: {receptor_pdbqt}")
    print(f"Centro: {center}")
    print(f"Box: {box_size}")

    results = [None] * len(tasks)

    ctx = get_context("spawn")
    with ctx.Pool(
        processes=n_workers,
        initializer=init_worker,
        initargs=(worker_cfg,),
    ) as pool:
        done = 0
        for idx, res in pool.imap_unordered(worker, tasks):
            results[idx] = res
            done += 1
            print(f"[{done}/{len(tasks)}] OK")

    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)

    print(f"Resultados guardados en: {output_csv_path}")


if __name__ == "__main__":
    main()