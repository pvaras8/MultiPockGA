# MultiPockGA

MultiPockGA is a multi-target molecular optimisation workflow. It uses
AutoGrow operators to generate molecules, AutoDock Vina to dock them against
multiple receptors, and RDKit properties such as QED to calculate fitness.

## Workflow

```text
SMILES
  -> AutoGrow crossover and mutation
  -> Gypsum-DL 3D ligand preparation
  -> MGLTools PDBQT conversion
  -> AutoDock Vina docking
  -> multi-target docking score + QED
  -> next GA generation
```

## Requirements

- Linux recommended
- Conda or Mamba
- Python 3.10
- MGLTools 1.5.6
- GNU `timeout` (`gtimeout` on macOS)

The project bundle includes AutoGrow, Gypsum-DL, and AutoDock Vina 1.1.2.

## Installation

```bash
git clone https://github.com/pvaras8/MultiPockGA.git
cd MultiPockGA

conda env create -f environment.yml
conda activate multipockga

bash scripts/setup_docking_vina.sh
export PYTHONPATH="$PWD/docking_vina${PYTHONPATH:+:$PYTHONPATH}"
```

Install MGLTools 1.5.6 separately from the
[official website](https://ccsb.scripps.edu/mgltools/).

If necessary, make the bundled Vina executable available:

```bash
chmod u+x docking_vina/autogrow/docking/docking_class/docking_executables/vina/autodock_vina_1_1_2_linux_x86/bin/vina
```

## Configuration

The main configuration is [config/docking.yaml](config/docking.yaml).

Each receptor has a separate JSON file:

- `config/vars_mgltools_6eif.json`
- `config/vars_mgltools_7q1p.json`
- `config/vars_mgltools_5tol.json`

These files contain paths from the original computing environment. Before
running the code, replace the following values with valid absolute paths:

```json
{
  "prepare_ligand4.py": "/path/to/prepare_ligand4.py",
  "prepare_receptor4.py": "/path/to/prepare_receptor4.py",
  "mgl_python": "/path/to/mgltools/bin/pythonsh",
  "filename_of_receptor": "/path/to/receptor_processed.pdb",
  "root_output_folder": "/path/to/MultiPockGA/docking_vina",
  "mgltools_directory": "/path/to/mgltools/"
}
```

The supplied docking boxes are:

| Receptor | Centre (x, y, z), A | Size (x, y, z), A |
| --- | --- | --- |
| 6EIF | 7.80, 17.01, -47.09 | 20, 20, 20 |
| 7Q1P | 18.44, 42.10, 43.10 | 20, 20, 20 |
| 5TOL | -27.66, 51.89, 20.25 | 20, 20, 20 |

Set the input file in `ga.source_compound_file`. CSV input requires the column
specified by `ga.source_smiles_column`; SMI and text files use the first field
of each line as the SMILES string.

For a DYRK1A/BuChE experiment, enable the 6EIF and 7Q1P providers and set:

```yaml
reward:
  combiner: docking_two_qed
  docking_columns: [Docking_6EIF, Docking_7Q1P]
  docking_alpha: 0.5
  docking_beta: 0.5
```

## Run

Check that the main packages are importable:

```bash
python -c "import rdkit, autogrow, multipockga; print('imports OK')"
```

Start the genetic algorithm:

```bash
python scripts/run_ga.py --config config/docking.yaml
```

To keep a complete log:

```bash
python scripts/run_ga.py --config config/docking.yaml 2>&1 | tee run.log
```

## Outputs

Results are written under the configured experiment directory:

```text
experiments/ga_run/
  population_epoch_<n>.csv
  generated_epoch_<n>.csv
  population_final.csv
  rewards/reward_results_<n>.csv
  tmp/<receptor>/docking_results_<n>_temp.csv
```

Temporary ligand and docking files are removed after each receptor run.

## Reproducibility notes

- Input CSV and SMI files, experimental results, and `docking_vina/` are ignored
  by Git. Archive them separately when reproducing an experiment.
- The original preparation procedure for the supplied `*_processed.pdb` files
  is not included in this repository.
- `docking_exhaustiveness` and `docking_num_modes` are currently `null`, so
  Vina uses its internal defaults.
- `random_seed` controls the Python genetic algorithm, not Vina or Gypsum-DL.
- Avoid simultaneous runs using the same receptor and output folder.
- Record the Git commit, configuration files, dependency versions, input data,
  processed receptors, and their SHA-256 checksums for each reported run.

## Tests

No automated tests are currently committed.

## Licence

No licence file is currently included.
