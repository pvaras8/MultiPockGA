import os
import shutil

import pandas as pd
import json
import random
import sys
from collections import defaultdict
import copy
from autogrow.user_vars import multiprocess_handling, determine_bash_timeout_vs_gtimeout
from autogrow.docking.execute_docking import pick_run_conversion_class_dict, pick_docking_class_dict, lig_convert_multithread
import autogrow.docking.scoring.execute_scoring_mol as Scoring
import autogrow.docking.ranking.ranking_mol as Ranking
import autogrow.operators.mutation.smiles_click_chem.smiles_click_chem as SmileClickClass
import autogrow.operators.convert_files.conversion_to_3d as conversion_to_3d
from autogrow.docking.execute_docking import run_dock_multithread, run_docking_common

# Leer archivo vars.json
with open(sys.argv[2], 'r') as f:
    vars = json.load(f)

# **Inicialización del paralelizador**
vars = multiprocess_handling(vars)

# Validación del sistema operativo para determinar el timeout
timeout_option = determine_bash_timeout_vs_gtimeout()
if timeout_option in ["timeout", "gtimeout"]:
    vars["timeout_vs_gtimeout"] = timeout_option
else:
    raise Exception("El sistema operativo no es compatible o falta Bash.")

# Inicialización de variables principales
smiles2info = defaultdict(dict)
id2smiles = dict()


# Función para actualizar información del receptor
def update_receptor_info(vars, receptor_info):
    name_of_receptor, filename_of_receptor, center_x, center_y, center_z, size_x, size_y, size_z = receptor_info
    vars['name_of_receptor'] = name_of_receptor
    vars['filename_of_receptor'] = filename_of_receptor
    vars['center_x'] = center_x
    vars['center_y'] = center_y
    vars['center_z'] = center_z
    vars['size_x'] = size_x
    vars['size_y'] = size_y
    vars['size_z'] = size_z
    return vars

def smiles_to_sdfs(vars, gen_smiles_file, smile_file_directory):
    # adapted from conversion_to_3d.convert_smi_to_sdfs_with_gypsum 
    max_variants_per_compound = vars["max_variants_per_compound"]
    gypsum_thoroughness = vars["gypsum_thoroughness"]
    min_ph = vars["min_ph"]
    max_ph = vars["max_ph"]
    pka_precision = vars["pka_precision"]
    gypsum_timeout_limit = vars["gypsum_timeout_limit"]

    # Make a new folder to put gypsum .smi's and json. Name folder gypsum_submission_files.
    folder_path = "{}gypsum_submission_files{}".format(smile_file_directory, os.sep)
    if os.path.exists(folder_path) is False:
        os.makedirs(folder_path)

    # Make Output for Gypsum folder (where .sdf's go)
    gypsum_output_folder_path = "{}_SDF{}".format(smile_file_directory, os.sep)
    if os.path.exists(gypsum_output_folder_path) is False:
        os.makedirs(gypsum_output_folder_path)

    # Make a folder to put the log files into within the 3D_SDFs folder
    gypsum_log_path = "{}log{}".format(gypsum_output_folder_path, os.sep)
    if os.path.exists(gypsum_log_path) is False:
        os.makedirs(gypsum_log_path)

    # Make All of the json files to submit to gypsum
    list_of_gypsum_params = conversion_to_3d.make_smi_and_gyspum_params(
        gen_smiles_file,
        folder_path,
        gypsum_output_folder_path,
        max_variants_per_compound, gypsum_thoroughness,
        min_ph, max_ph, pka_precision, )

    # create a the job_inputs to run gypsum in multithread
    job_input = tuple([(gypsum_log_path, gypsum_params, gypsum_timeout_limit) for gypsum_params in list_of_gypsum_params])

    sys.stdout.flush()
    failed_to_convert = vars["parallelizer"].run(job_input, conversion_to_3d.run_gypsum_multiprocessing)
    sys.stdout.flush()

    ###    fail: return smiles 
    ###    success: return None     
    lig_failed_to_convert = [x for x in failed_to_convert if x is not None]
    lig_failed_to_convert = list(set(lig_failed_to_convert))
    if len(lig_failed_to_convert) > 0:
        print("The Following ligands Failed to convert in Gypsum")
        print("Likely due to a Timeout")
        print(lig_failed_to_convert)
    sys.stdout.flush()
    return gypsum_output_folder_path

def pdb_to_pdbqt(vars, pdb_dir):
    ### adapted from run_docking_common
    dock_choice = vars["dock_choice"]
    conversion_choice = vars["conversion_choice"]
    receptor = vars["filename_of_receptor"]
    print('El receptor es', receptor)

    # Use a temp vars dict so you don't put mpi multiprocess info through itself...
    temp_vars = {}
    for key in list(vars.keys()):
        if key == "parallelizer":
            continue
        temp_vars[key] = vars[key]

    file_conversion_class_object = pick_run_conversion_class_dict(conversion_choice)
    file_conversion_class_object = file_conversion_class_object(temp_vars, receptor, test_boot=False)

    dock_class = pick_docking_class_dict(dock_choice)
    docking_object = dock_class(temp_vars, receptor, file_conversion_class_object, test_boot=False)

    if vars["docking_executable"] is None:
        docking_executable = docking_object.get_docking_executable_file(temp_vars)
        vars["docking_executable"] = docking_executable
    ##### vina or Qvina 

    # Find PDB's
    pdbs_in_folder = docking_object.find_pdb_ligands(pdb_dir)
    
    # Convert relative paths to absolute paths
    pdbs_absolute = [os.path.abspath(pdb) for pdb in pdbs_in_folder]
    print('    pdb files (absolute paths):', pdbs_absolute[:2], os.path.abspath(pdb_dir), len(pdbs_absolute))
    
    job_input_convert_lig = tuple([(docking_object, pdb) for pdb in pdbs_absolute])

    # print("    Convert Ligand from PDB to PDBQT format")
    smiles_names_failed_to_convert = vars["parallelizer"].run(job_input_convert_lig, lig_convert_multithread)

    pdbqts_in_folder = docking_object.find_converted_ligands(pdb_dir)
    pdbqts_absolute = [os.path.abspath(pdbqt) for pdbqt in pdbqts_in_folder]
    print('    pdbqt file (absolute paths):', len(pdbqts_absolute), pdbqts_absolute[:2])
    
    return docking_object

def docking_pdbqt(vars, docking_object, pdbqt_folder, full_smiles_file):
    pdbqts_in_folder = docking_object.find_converted_ligands(pdbqt_folder)
    job_input_dock_lig = tuple([tuple([docking_object, pdbqt]) for pdbqt in pdbqts_in_folder])
    smiles_names_failed_to_dock = vars["parallelizer"].run(job_input_dock_lig, run_dock_multithread)  
    ### main docking, (including delete failed docking file)

    deleted_smiles_names_list_dock = [x for x in smiles_names_failed_to_dock if x is not None]
    deleted_smiles_names_list_dock = list(set(deleted_smiles_names_list_dock))
    print("THE FOLLOWING LIGANDS WHICH FAILED TO DOCK:", deleted_smiles_names_list_dock)
    # print("#################### \n Begin Ranking and Saving results")
    # folder_with_pdbqts = current_generation_dir + "PDBs" + os.sep
    # Run any compatible Scoring Function
    smiles_list = Scoring.run_scoring_common(vars, full_smiles_file, pdbqt_folder)
    print (smiles_list[:3])
    return smiles_list
# Función principal para el pipeline de docking
def docking(smiles_folder, smiles_file, vars):
    print(smiles_folder)
    sdfs_folder_path = smiles_folder + '_SDF/'
    print(sdfs_folder_path)
    pdb_dir = smiles_folder + '_PDB/'
    print(pdb_dir)
    gypsum_submission_dir = "{}gypsum_submission_files{}".format(smiles_folder, os.sep)

    try:
        smiles_to_sdfs(vars, gen_smiles_file=os.path.join(smiles_folder, smiles_file), smile_file_directory=smiles_folder)
        conversion_to_3d.convert_sdf_to_pdbs(vars, gen_folder_path=smiles_folder, sdfs_folder_path=sdfs_folder_path)
        docking_object = pdb_to_pdbqt(vars, pdb_dir=pdb_dir)
        smiles_list = docking_pdbqt(vars, docking_object, pdb_dir, os.path.join(smiles_folder, smiles_file))
        return smiles_list
    finally:
        # Limpiar carpetas temporales de la generacion
        shutil.rmtree(pdb_dir, ignore_errors=True)
        shutil.rmtree(sdfs_folder_path, ignore_errors=True)
        shutil.rmtree(gypsum_submission_dir, ignore_errors=True)
        shutil.rmtree(smiles_folder, ignore_errors=True)

# Actualiza el archivo fuente (SMILES)
source_compound_file = sys.argv[1]
epoch = sys.argv[3]             # Época pasada como argumento

vars["source_compound_file"] = source_compound_file


def infer_receptor_name(receptor_path):
    receptor_base = os.path.basename(receptor_path)
    receptor_stem = os.path.splitext(receptor_base)[0]
    if receptor_stem.endswith("_processed"):
        return receptor_stem[: -len("_processed")]
    return receptor_stem

# Información del receptor
receptor_info_list = [
    (
        vars.get('name_of_receptor', infer_receptor_name(vars['filename_of_receptor'])),
        vars['filename_of_receptor'],
        vars['center_x'],
        vars['center_y'],
        vars['center_z'],
        vars['size_x'],
        vars['size_y'],
        vars['size_z'],
    )
]

# Lectura de SMILES iniciales desde el archivo generado
with open(source_compound_file, 'r') as fin:
    initial_smiles_list = [line.split()[0] for line in fin]


# Pipeline principal
print("---------- 1. Prediccion Docking ----------")
final_results = []  # Lista para almacenar los resultados finales
for receptor_info in receptor_info_list:
    vars = update_receptor_info(vars, receptor_info)
    name_of_receptor = receptor_info[0]

    # Directorios de resultados
    meta_result_folder = f'{vars["root_output_folder"]}/results_{name_of_receptor}_'
    results_folder = f'{meta_result_folder}000'
    print(f"Intentando crear el directorio en: {results_folder}")
    os.makedirs(results_folder, exist_ok=True)

    # Archivo de SMILES
    full_smiles_file = os.path.join(results_folder, 'smiles.txt')

    with open(full_smiles_file, 'w') as fout:
        for idx, smiles in enumerate(initial_smiles_list, start=1):  # start=1 para iniciar desde 1
            smiles_id = str(idx)  # ID secuencial basado en el índice
            fout.write(f'{smiles}\t{smiles_id}\n')
            id2smiles[smiles_id] = smiles


    # Ejecutar docking
    smiles_list = docking(smiles_folder=results_folder, smiles_file='smiles.txt', vars=vars)
    print('this is the final', smiles_list)

# Crear un diccionario con los resultados del docking exitoso
docking_scores = {info[1]: float(info[-1]) for info in smiles_list}  # ID como clave





# Procesar todos los SMILES, incluyendo aquellos que fallaron
for idx, smiles in enumerate(initial_smiles_list, start=1):
    smiles_id = str(idx)  # ID secuencial
    docking_score = docking_scores.get(smiles_id, -6.0)  # Si falla, asignar -6
    smiles2info[name_of_receptor][smiles] = [docking_score]
    final_results.append({"ID": smiles_id, "SMILES": smiles, "Docking": docking_score})

# Ordenar los resultados por ID
final_results_sorted = sorted(final_results, key=lambda x: int(x['ID']))




# Guardar resultados en un archivo CSV
final_folder = vars.get("final_folder")
if not final_folder:
    final_folder = os.path.join(vars.get("root_output_folder", os.getcwd()), "reward_outputs")
os.makedirs(final_folder, exist_ok=True)
intermediate_csv_path = os.path.join(final_folder, f'docking_results_{epoch}_temp.csv')

pd.DataFrame(final_results_sorted).to_csv(intermediate_csv_path, index=False)
print(f"Resultados intermedios guardados en: {intermediate_csv_path}")




