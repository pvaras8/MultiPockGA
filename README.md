# MultiPockGA

MultiPockGA is a multi-target molecular optimisation workflow that combines a
genetic algorithm with structure-based docking and RDKit molecular properties.
Candidate molecules are generated with AutoGrow crossover and mutation
operators, docked independently against each configured receptor with AutoDock
Vina, and ranked with a configurable reward function.

The repository contains the orchestration code and configuration files.
AutoGrow, Gypsum-DL, receptor structures, MGLTools, and the input dataset are
external runtime requirements; read the reproducibility notes below before
starting a run.

## Workflow

For every generation, MultiPockGA:

1. reads and canonicalises an initial set of SMILES with RDKit;
2. generates crossover and mutation candidates with AutoGrow;
3. filters invalid or duplicate candidates;
4. prepares ligand variants and three-dimensional conformers with Gypsum-DL;
5. converts ligands and receptors to PDBQT with MGLTools;
6. docks every valid ligand independently against each configured receptor;
7. extracts the most negative Vina score across poses and ligand variants;
8. computes RDKit properties, including QED; and
9. combines the target scores and QED into a fitness value used to select the
   next population.

## Repository layout

```text
config/                  GA, reward, receptor, and docking configuration
multipockga/             Python package
  ga/                    genetic-algorithm runner
  rewards/               reward providers and fitness combiners
scripts/                 setup and execution entry points
datasets/                local input datasets; CSV/SMI files are ignored by Git
experiments/             generated populations, docking tables, and logs
```

## Requirements

- Linux is recommended for production runs.
- Conda or Mamba.
- Python 3.10.
- MGLTools 1.5.6, installed separately.
- GNU `timeout` on Linux, or GNU `gtimeout` on macOS.
- Sufficient CPU and storage for the selected population size and receptors.

The downloaded AutoGrow bundle includes Gypsum-DL and AutoDock Vina 1.1.2.
Windows is not supported by the bundled Vina launcher.

## Installation

Clone the repository and create the environment:

```bash
git clone https://github.com/pvaras8/MultiPockGA.git
cd MultiPockGA
conda env create -f environment.yml
conda activate multipockga
```

Download the external AutoGrow/Vina bundle used by the project:

```bash
bash scripts/setup_docking_vina.sh
```

The bundle is placed in `docking_vina/`, which is intentionally ignored by
Git. Ensure that the packaged docking executables are executable after
extraction:

```bash
chmod u+x \
  docking_vina/autogrow/docking/docking_class/docking_executables/vina/autodock_vina_1_1_2_linux_x86/bin/vina \
  docking_vina/autogrow/docking/docking_class/docking_executables/q_vina_2/q_vina_2_1_linux/vina
```

Make AutoGrow importable in the current shell:

```bash
export PYTHONPATH="$PWD/docking_vina${PYTHONPATH:+:$PYTHONPATH}"
```

Install MGLTools 1.5.6 manually from the
[MGLTools website](https://ccsb.scripps.edu/mgltools/). Confirm that these files
exist in the installation:

```text
bin/pythonsh
MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py
MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py
```

## Runtime configuration

The checked-in receptor JSON files contain paths from the original computing
environment. A fresh clone will not run until these paths are replaced with
paths valid on the current machine.

For every receptor JSON used by `config/docking.yaml`, update:

```json
{
  "prepare_ligand4.py": "/absolute/path/to/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py",
  "prepare_receptor4.py": "/absolute/path/to/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py",
  "mgl_python": "/absolute/path/to/mgltools/bin/pythonsh",
  "filename_of_receptor": "/absolute/path/to/receptor_processed.pdb",
  "root_output_folder": "/absolute/path/to/MultiPockGA/docking_vina",
  "mgltools_directory": "/absolute/path/to/mgltools/"
}
```

The downloaded bundle provides processed receptor files under
`docking_vina/pdb/`. The supplied target configurations are:

| File | Receptor | Box centre (x, y, z), A | Box size (x, y, z), A |
| --- | --- | --- | --- |
| `config/vars_mgltools_6eif.json` | 6EIF | 7.80, 17.01, -47.09 | 20, 20, 20 |
| `config/vars_mgltools_7q1p.json` | 7Q1P | 18.44, 42.10, 43.10 | 20, 20, 20 |
| `config/vars_mgltools_5tol.json` | 5TOL | -27.66, 51.89, 20.25 | 20, 20, 20 |

Receptor preparation before the creation of the supplied `*_processed.pdb`
files is not implemented in this repository. For a reproducible new receptor,
record the source PDB revision and independently document chain selection,
water and heteroatom removal, missing-residue treatment, protonation, and any
other cleaning operations.

### Input dataset

Set `ga.source_compound_file` in `config/docking.yaml` to a local `.csv`, `.smi`,
or text file. For CSV input, `ga.source_smiles_column` names the SMILES column.
For SMI or plain-text input, the first whitespace-separated field on each
non-empty line is interpreted as a SMILES string.

CSV and SMI files are ignored by the repository-wide `.gitignore`. They are
therefore not guaranteed to be present in a fresh clone. To reproduce a run,
archive the exact input file separately and record its checksum, for example:

```bash
shasum -a 256 /path/to/input.csv
```

### Selecting targets

Targets are entries under `reward.providers` in `config/docking.yaml`. Each
active docking provider must have a unique output column, and
`reward.docking_columns` must contain the two columns consumed by the selected
multi-target combiner.

For a DYRK1A/BuChE run, enable the 6EIF and 7Q1P providers and use:

```yaml
reward:
  combiner: docking_two_qed
  docking_columns: [Docking_6EIF, Docking_7Q1P]
  docking_alpha: 0.5
  docking_beta: 0.5
```

The checked-in configuration may target a different receptor pair. Always
record the exact YAML and receptor JSON files with the results of an experiment.

## Running an experiment

Validate the environment first:

```bash
python -c "import rdkit, autogrow, multipockga; print('imports OK')"
command -v timeout || command -v gtimeout
```

Run the genetic algorithm:

```bash
python scripts/run_ga.py --config config/docking.yaml
```

The main GA settings are controlled in the `ga` section of the YAML file:

- `num_generations`: number of evolutionary generations after generation 0;
- `top_k`: maximum retained population size;
- `crossover_budget` and `mutation_budget`: numbers of parents sampled for
  each operator;
- `random_seed`: Python random seed used by the GA; and
- `maximize_fitness`: whether larger fitness values are preferred.

The GA seed does not set the random seed used by Vina or Gypsum-DL. The current
docking configuration also leaves `docking_exhaustiveness` and
`docking_num_modes` as `null`, so the packaged Vina executable uses its own
defaults. Set these fields explicitly in every receptor JSON when exact docking
settings must be recorded.

## Outputs

With the default output layout, a run produces:

```text
experiments/ga_run/
  autogrow_source_from_csv.smi
  population_epoch_<n>.csv
  generated_epoch_<n>.csv
  population_final.csv
  rewards/
    reward_results_<n>.csv
  tmp/
    <receptor>/
      docking_results_<n>_temp.csv
```

The combined reward tables contain the input index, input SMILES, per-target
docking scores, RDKit properties, and fitness. Temporary Gypsum-DL, SDF, PDB,
PDBQT, and Vina pose directories are removed after each receptor run. Keep the
console log if preparation failures and timeouts must be audited:

```bash
python scripts/run_ga.py --config config/docking.yaml 2>&1 | tee run.log
```

Do not launch two runs that share the same `root_output_folder` and receptor at
the same time. The docking script uses the fixed directory name
`results_<receptor>_000`, so concurrent runs can overwrite or remove one
another's intermediate files.

## Docking and fitness defaults

The supplied MGLTools receptor configurations use:

- `MGLToolsConversion`;
- AutoDock Vina 1.1.2 with one CPU per ligand;
- a 1,800 s docking timeout;
- one Gypsum-DL variant per input compound;
- Gypsum-DL thoroughness 3;
- an ionisation range of pH 6.4--8.4;
- a 300 s ligand-preparation timeout; and
- a failed-docking fallback score of -6.0.

For `docking_two_qed`, a docking score `d` is transformed as follows:

```text
z(d) = -6 + ((d + 12) / 6) * 12
r(d) = 1 - 1 / (1 + exp(-z(d)))
```

The same `[-12, -6]` score range is used for both targets. The two rewards are
combined by a weighted geometric mean. QED is transformed using a sigmoid with
threshold 0.5, sharpness 10, and floor 0.2. The product of the docking and QED
components is clipped to `[0.02, 1.0]`.

## Error handling

- Ligand preparation failures are reported and omitted from docking.
- MGLTools ligand conversion is attempted up to three times.
- Docking failures without a valid score receive -6.0.
- Provider outputs are checked for row count and SMILES order.
- With `reward.fail_on_error: true`, unrecoverable provider or file errors stop
  the run instead of silently assigning zero fitness to the full population.

## Tests

When tests are added under `tests/`, run them with:

```bash
pytest -q
```

At present, no automated tests are committed, and the repository does not
include an end-to-end fixture containing an input dataset, receptors, MGLTools,
and expected docking output. Consequently, the full docking workflow must be
validated from its generated logs and result tables.

## Reproducibility checklist

For every reported experiment, retain or record:

- the MultiPockGA Git commit;
- the exact YAML and receptor JSON files;
- the input dataset and its SHA-256 checksum;
- the processed receptor PDB and PDBQT files and their checksums;
- the receptor-preparation procedure;
- the MGLTools, AutoGrow/Gypsum-DL, RDKit, and operating-system versions;
- explicit Vina exhaustiveness, number of modes, and seed if required;
- the available CPU count represented by `number_of_processors: -1`; and
- the complete console log and per-generation reward tables.

Useful commands are:

```bash
git rev-parse HEAD
conda env export --no-builds > environment.lock.yml
shasum -a 256 config/docking.yaml config/vars_mgltools_*.json
```

## Licence and citation

No licence or citation metadata is currently included in the repository. Add a
licence before redistributing the code, and cite AutoGrow, Gypsum-DL,
AutoDockTools/MGLTools, AutoDock Vina, and RDKit as appropriate in scientific
work.
