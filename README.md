# MultiPockGA

Proyecto base para experimentos con algoritmos geneticos.

## Estructura

- `multipockga/`: paquete principal
- `config/`: configuraciones
- `datasets/`: datos de entrada
- `scripts/`: scripts utilitarios y ejecucion
- `experiments/`: notebooks/resultados/pruebas experimentales
- `tests/`: tests automatizados

## Inicio rapido

### Conda

```bash
conda env create -f environment.yml
conda activate multipockga
```

### Pip

```bash
pip install -e .
```

### Tests

```bash
pytest -q
```

## Seleccion De Docking

El GA usa siempre operadores de AutoGrow para crossover y mutation.

- Para docking con Vina/MGLTools usa [config/docking.yaml](config/docking.yaml).
- Para docking con Meeko+Vina usa [config/docking_meeko.yaml](config/docking_meeko.yaml).

Ejecucion:

```bash
python scripts/run_ga.py -c config/docking.yaml
python scripts/run_ga.py -c config/docking_meeko.yaml
```
