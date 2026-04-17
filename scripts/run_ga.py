import argparse
import os

import yaml

from multipockga.ga import GARunner


def load_config(config_path: str) -> dict:
	abs_cfg_path = os.path.abspath(config_path)
	config_dir = os.path.dirname(abs_cfg_path)

	with open(abs_cfg_path, "r") as f:
		cfg = yaml.safe_load(f)

	if not isinstance(cfg, dict):
		raise ValueError("El archivo de configuracion debe contener un objeto YAML")

	cfg["_config_path"] = abs_cfg_path
	cfg["_config_dir"] = config_dir
	return cfg


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Ejecuta el pipeline GA de MultiPockGA usando configuracion YAML",
	)
	parser.add_argument(
		"--config",
		"-c",
		type=str,
		default="config/docking.yaml",
		help="Ruta al archivo YAML de configuracion",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	cfg = load_config(args.config)
	runner = GARunner(cfg)
	runner.run()


if __name__ == "__main__":
	main()
