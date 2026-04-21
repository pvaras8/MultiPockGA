import json
import os
import random
from typing import Dict, List, Set

import pandas as pd
from rdkit import Chem

from multipockga.rewards import RewardRunner


class GARunner:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.config_dir = os.path.abspath(self.cfg.get("_config_dir", os.getcwd()))
        self.repo_root = os.path.abspath(
            self.cfg.get("_repo_root", os.path.join(self.config_dir, os.pardir))
        )

        self.ga_cfg = cfg.get("ga", {})
        self.autogrow_cfg = cfg.get("autogrow", {})

        self.num_generations = int(self.ga_cfg.get("num_generations", 1))
        self.top_k = int(self.ga_cfg.get("top_k", 100))
        self.crossover_budget = int(self.ga_cfg.get("crossover_budget", 20))
        self.mutation_budget = int(self.ga_cfg.get("mutation_budget", 20))
        self.crossover_attempts = int(self.ga_cfg.get("crossover_attempts", 3))
        self.mutation_children_limit = int(self.ga_cfg.get("mutation_children_limit", 0))
        self.operator_backend = str(self.ga_cfg.get("operator_backend", "autogrow")).lower()
        self.crossover_backend = str(
            self.ga_cfg.get("crossover_backend", self.operator_backend)
        ).lower()
        self.mutation_backend = str(
            self.ga_cfg.get("mutation_backend", self.operator_backend)
        ).lower()
        self.crossover_products_per_pair = int(
            self.ga_cfg.get("crossover_products_per_pair", 4)
        )
        self.selfies_mutation_tries = int(self.ga_cfg.get("selfies_mutation_tries", 30))
        self.maximize_fitness = bool(self.ga_cfg.get("maximize_fitness", True))

        self.output_dir = self._resolve_path(self.ga_cfg.get("output_dir", "experiments/ga_run"))
        os.makedirs(self.output_dir, exist_ok=True)

        seed = self.ga_cfg.get("random_seed")
        if seed is not None:
            random.seed(int(seed))

        self.reward_runner = RewardRunner(cfg)
        self.vars = self._build_autogrow_vars()

        self._bootstrap_autogrow_objects()
        self.seen_smiles: Set[str] = set()

    def _resolve_path(self, path_value: str) -> str:
        if os.path.isabs(path_value):
            return path_value

        if path_value.startswith("./") or path_value.startswith("../"):
            return os.path.abspath(os.path.join(self.config_dir, path_value))

        return os.path.abspath(os.path.join(self.repo_root, path_value))

    def _build_autogrow_vars(self) -> Dict:
        if "vars_file" not in self.autogrow_cfg:
            raise ValueError("Falta autogrow.vars_file en el config")

        vars_file = self._resolve_path(self.autogrow_cfg["vars_file"])
        with open(vars_file, "r") as f:
            vars_dict = json.load(f)

        for key, value in self.autogrow_cfg.items():
            if key == "vars_file":
                continue
            if key == "overrides" and isinstance(value, dict):
                vars_dict.update(value)
                continue
            vars_dict[key] = value

        self._ensure_autogrow_source_compound_file(vars_dict)

        from autogrow.user_vars import (
            determine_bash_timeout_vs_gtimeout,
            load_in_commandline_parameters,
            multiprocess_handling,
        )

        inputs = {k: v for k, v in vars_dict.items() if v is not None}
        args_dict, _ = load_in_commandline_parameters(inputs)
        args_dict = multiprocess_handling(args_dict)

        timeout_option = determine_bash_timeout_vs_gtimeout()
        if timeout_option in ["timeout", "gtimeout"]:
            args_dict["timeout_vs_gtimeout"] = timeout_option
        else:
            raise RuntimeError("No se pudo determinar timeout/gtimeout para Autogrow")

        return args_dict

    def _ensure_autogrow_source_compound_file(self, vars_dict: Dict) -> None:
        source_smiles_column = self.ga_cfg.get("source_smiles_column", "SMILES")

        # Priorizamos la fuente de GA (csv/smi) por sobre la heredada de vars.
        candidate_source = self.ga_cfg.get("source_compound_file") or vars_dict.get(
            "source_compound_file"
        )
        if not candidate_source:
            raise ValueError("No se encontro source_compound_file en ga ni en vars de autogrow")

        source_path = self._resolve_path(candidate_source)
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"No existe source_compound_file: {source_path}")

        if source_path.lower().endswith(".smi"):
            vars_dict["source_compound_file"] = source_path
            return

        if source_path.lower().endswith(".csv"):
            df = pd.read_csv(source_path)
            if source_smiles_column not in df.columns:
                raise ValueError(
                    f"El CSV inicial no contiene la columna '{source_smiles_column}'. "
                    f"Columnas disponibles: {list(df.columns)}"
                )

            temp_smi_path = os.path.join(self.output_dir, "autogrow_source_from_csv.smi")
            written = 0
            with open(temp_smi_path, "w") as fout:
                for idx, smi in enumerate(df[source_smiles_column].astype(str).tolist(), start=1):
                    smi_canon = self._canonicalize(smi)
                    if not smi_canon:
                        continue
                    fout.write(f"{smi_canon}\tseed_{idx}\n")
                    written += 1

            if written == 0:
                raise ValueError(
                    "No se pudieron convertir SMILES validos desde el CSV para crear el .smi de AutoGrow"
                )

            vars_dict["source_compound_file"] = temp_smi_path
            return

        # Si viene en otro formato de texto, intentamos interpretarlo como lista de smiles.
        temp_smi_path = os.path.join(self.output_dir, "autogrow_source_from_text.smi")
        written = 0
        with open(source_path, "r") as fin, open(temp_smi_path, "w") as fout:
            for idx, line in enumerate(fin, start=1):
                line = line.strip()
                if not line:
                    continue
                smi = line.split()[0]
                smi_canon = self._canonicalize(smi)
                if not smi_canon:
                    continue
                fout.write(f"{smi_canon}\tseed_{idx}\n")
                written += 1

        if written == 0:
            raise ValueError(
                f"No se pudieron extraer SMILES validos desde source_compound_file: {source_path}"
            )

        vars_dict["source_compound_file"] = temp_smi_path

    def _bootstrap_autogrow_objects(self) -> None:
        self._smiles_click_chem_cls = None
        self.rxn_library_variables = []

        if self.mutation_backend != "autogrow":
            return

        import autogrow.operators.mutation.smiles_click_chem.smiles_click_chem as SmileClickClass

        self._smiles_click_chem_cls = SmileClickClass.SmilesClickChem
        self.rxn_library_variables = [
            self.vars["rxn_library"],
            self.vars["rxn_library_file"],
            self.vars["function_group_library"],
            self.vars["complementary_mol_directory"],
        ]

    @staticmethod
    def _canonicalize(smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol, isomericSmiles=True)

    def _load_initial_smiles(self) -> List[str]:
        source_file = self.ga_cfg.get("source_compound_file") or self.vars.get("source_compound_file")
        if source_file is None:
            raise ValueError("Falta ga.source_compound_file o source_compound_file en vars")

        source_file = self._resolve_path(source_file)
        source_smiles_column = self.ga_cfg.get("source_smiles_column", "SMILES")
        smiles = []

        if source_file.lower().endswith(".csv"):
            df = pd.read_csv(source_file)
            if source_smiles_column not in df.columns:
                raise ValueError(
                    f"El CSV inicial no contiene la columna '{source_smiles_column}'. "
                    f"Columnas disponibles: {list(df.columns)}"
                )

            for smi in df[source_smiles_column].astype(str).tolist():
                smi_canon = self._canonicalize(smi)
                if smi_canon:
                    smiles.append(smi_canon)
        else:
            with open(source_file, "r") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    smi = line.split()[0]
                    smi_canon = self._canonicalize(smi)
                    if smi_canon:
                        smiles.append(smi_canon)

        if len(smiles) == 0:
            raise ValueError("No se encontraron SMILES validos en source_compound_file")

        if len(smiles) > self.top_k:
            smiles = smiles[: self.top_k]
        return smiles

    def _passes_filters(self, smiles: str) -> bool:
        import autogrow.operators.filter.execute_filters as Filter

        return bool(Filter.run_filter_on_just_smiles(smiles, self.vars["filter_object_dict"]))

    def _generate_crossover_candidates(self, smiles_list: List[str]) -> Set[str]:
        if self.crossover_backend == "autogrow":
            return self._generate_crossover_candidates_autogrow(smiles_list)
        if self.crossover_backend in {"brics", "brics_selfies", "simple"}:
            return self._generate_crossover_candidates_brics(smiles_list)
        raise ValueError(f"crossover backend desconocido: {self.crossover_backend}")

    def _generate_crossover_candidates_autogrow(self, smiles_list: List[str]) -> Set[str]:
        import autogrow.operators.crossover.execute_crossover as execute_crossover
        import autogrow.operators.crossover.smiles_merge.smiles_merge as smiles_merge

        if len(smiles_list) < 2 or self.crossover_budget <= 0:
            return set()

        chosen = random.sample(smiles_list, k=min(self.crossover_budget, len(smiles_list)))
        generated: Set[str] = set()
        done_pairs = set()

        for smi_1 in chosen:
            mol_1 = execute_crossover.convert_mol_from_smiles(smi_1)
            if mol_1 is None:
                continue

            partners = [s for s in smiles_list if s != smi_1]
            random.shuffle(partners)
            for smi_2 in partners[: min(len(partners), self.crossover_budget)]:
                if (smi_1, smi_2) in done_pairs:
                    continue
                done_pairs.add((smi_1, smi_2))
                done_pairs.add((smi_2, smi_1))

                mol_2 = execute_crossover.convert_mol_from_smiles(smi_2)
                if mol_2 is None:
                    continue

                if execute_crossover.test_for_mcs(self.vars, mol_1, mol_2) is None:
                    continue

                candidate = None
                for _ in range(self.crossover_attempts):
                    candidate = smiles_merge.run_main_smiles_merge(self.vars, smi_1, smi_2)
                    if candidate:
                        break

                if not candidate:
                    continue

                candidate = self._canonicalize(candidate)
                if not candidate:
                    continue
                if candidate in self.seen_smiles:
                    continue
                if not self._passes_filters(candidate):
                    continue

                generated.add(candidate)

        return generated

    def _generate_mutation_candidates(self, smiles_list: List[str]) -> Set[str]:
        if self.mutation_backend == "autogrow":
            return self._generate_mutation_candidates_autogrow(smiles_list)
        if self.mutation_backend in {"selfies", "brics_selfies", "simple"}:
            return self._generate_mutation_candidates_selfies(smiles_list)
        raise ValueError(f"mutation backend desconocido: {self.mutation_backend}")

    def _generate_crossover_candidates_brics(self, smiles_list: List[str]) -> Set[str]:
        from multipockga.ga.crossover_rdkit import generate_crossovers

        if len(smiles_list) < 2 or self.crossover_budget <= 0:
            return set()

        generated = generate_crossovers(
            smiles_list,
            budget=self.crossover_budget,
            max_products_per_pair=self.crossover_products_per_pair,
            seed=self.ga_cfg.get("random_seed"),
        )

        filtered: Set[str] = set()
        for candidate in generated:
            candidate = self._canonicalize(candidate)
            if not candidate:
                continue
            if candidate in self.seen_smiles:
                continue
            if not self._passes_filters(candidate):
                continue
            filtered.add(candidate)

        return filtered

    def _generate_mutation_candidates_autogrow(self, smiles_list: List[str]) -> Set[str]:
        if len(smiles_list) == 0 or self.mutation_budget <= 0:
            return set()

        selected = random.sample(smiles_list, k=min(self.mutation_budget, len(smiles_list)))
        generated: Set[str] = set()

        for smi in selected:
            click_obj = self._smiles_click_chem_cls(
                rxn_library_variables=self.rxn_library_variables,
                list_of_already_made_smiles=[],
                filter_object_dict=self.vars["filter_object_dict"],
            )
            mutated = click_obj.run_smiles_click2(smi)
            if not mutated:
                continue

            if self.mutation_children_limit > 0:
                mutated = mutated[: self.mutation_children_limit]

            for cand in mutated:
                cand = self._canonicalize(cand)
                if not cand:
                    continue
                if cand in self.seen_smiles:
                    continue
                if not self._passes_filters(cand):
                    continue
                generated.add(cand)

        return generated

    def _generate_mutation_candidates_selfies(self, smiles_list: List[str]) -> Set[str]:
        from multipockga.ga.mutation_rdkit import generate_mutations

        if len(smiles_list) == 0 or self.mutation_budget <= 0:
            return set()

        generated = generate_mutations(
            smiles_list,
            budget=self.mutation_budget,
            seed=self.ga_cfg.get("random_seed"),
        )

        filtered: Set[str] = set()
        for cand in generated:
            cand = self._canonicalize(cand)
            if not cand:
                continue
            if cand in self.seen_smiles:
                continue
            if not self._passes_filters(cand):
                continue
            filtered.add(cand)

        return filtered

    def _evaluate_population(self, smiles_list: List[str], epoch: int) -> pd.DataFrame:
        rewards = self.reward_runner(smiles_list, epoch)
        df = pd.DataFrame({"SMILES": smiles_list, "Fitness": rewards})
        return self._sort_population(df)

    def _sort_population(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values("Fitness", ascending=not self.maximize_fitness).reset_index(drop=True)

    def _select_top_k(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._sort_population(df)
        df = df.drop_duplicates(subset=["SMILES"], keep="first")
        return df.head(self.top_k).reset_index(drop=True)

    def _save_epoch_population(self, df: pd.DataFrame, epoch: int) -> None:
        out_file = os.path.join(self.output_dir, f"population_epoch_{epoch}.csv")
        df.to_csv(out_file, index=False)

    def _save_generation_details(
        self,
        epoch: int,
        crossover_set: Set[str],
        mutation_set: Set[str],
    ) -> None:
        if epoch <= 0:
            return

        generated_smiles = sorted(crossover_set | mutation_set)
        rows = []
        for smi in generated_smiles:
            via_crossover = smi in crossover_set
            via_mutation = smi in mutation_set
            if via_crossover and via_mutation:
                origin = "crossover+mutation"
            elif via_crossover:
                origin = "crossover"
            else:
                origin = "mutation"
            rows.append({"SMILES": smi, "GeneratedBy": origin})

        generated_df = pd.DataFrame(rows)
        reward_df = self.reward_runner.last_epoch_results
        if reward_df is not None and int(self.reward_runner.last_epoch) == int(epoch):
            cols_to_keep = [c for c in reward_df.columns if c not in {"input_idx"}]
            generated_df = generated_df.merge(
                reward_df[cols_to_keep], on="SMILES", how="left"
            )

        out_file = os.path.join(self.output_dir, f"generated_epoch_{epoch}.csv")
        generated_df.to_csv(out_file, index=False)

    def run(self) -> pd.DataFrame:
        print("---------- GA configuration ----------")
        print(f"Output dir: {self.output_dir}")
        print(f"Operator backend: {self.operator_backend}")
        print(f"Crossover backend: {self.crossover_backend}")
        print(f"Mutation backend: {self.mutation_backend}")
        print(f"Generations: {self.num_generations}")
        print(f"Top-k: {self.top_k}")
        print(f"Crossover budget: {self.crossover_budget}")
        print(f"Mutation budget: {self.mutation_budget}")
        print(f"Reward output dir: {self.reward_runner.output_dir}")

        initial_smiles = self._load_initial_smiles()
        self.seen_smiles.update(initial_smiles)

        current_df = self._evaluate_population(initial_smiles, epoch=0)
        current_df = self._select_top_k(current_df)
        self._save_epoch_population(current_df, epoch=0)

        for epoch in range(1, self.num_generations + 1):
            parent_smiles = current_df["SMILES"].tolist()

            crossover_set = self._generate_crossover_candidates(parent_smiles)
            mutation_seed = list(crossover_set) if len(crossover_set) > 0 else parent_smiles
            mutation_set = self._generate_mutation_candidates(mutation_seed)

            print(f"---------- Generation {epoch} ----------")
            print(f"Molecules generated by crossover: {len(crossover_set)}")
            print(f"Molecules generated by mutation: {len(mutation_set)}")

            candidate_smiles = list(crossover_set | mutation_set)
            if len(candidate_smiles) == 0:
                self._save_generation_details(epoch, crossover_set, mutation_set)
                self._save_epoch_population(current_df, epoch=epoch)
                continue

            self.seen_smiles.update(candidate_smiles)

            candidate_df = self._evaluate_population(candidate_smiles, epoch=epoch)
            self._save_generation_details(epoch, crossover_set, mutation_set)
            merged_df = pd.concat([current_df, candidate_df], ignore_index=True)
            current_df = self._select_top_k(merged_df)
            self._save_epoch_population(current_df, epoch=epoch)

        final_file = os.path.join(self.output_dir, "population_final.csv")
        current_df.to_csv(final_file, index=False)
        return current_df
