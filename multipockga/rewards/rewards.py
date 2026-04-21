from typing import Dict, List, Optional, Tuple

import os

import pandas as pd
import selfies as sf

from .combiners import COMBINERS
from .providers.docking import DockingProvider
from .providers.rdkit_props import RDKitPropsProvider


PROVIDER_MAP = {
    "docking": DockingProvider,
    "rdkit_props": RDKitPropsProvider,
}


class RewardRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reward_cfg = cfg["reward"]
        self.input_type = self.reward_cfg.get("input_type", "smiles").lower()

        self.combiner_name = self.reward_cfg["combiner"]
        if self.combiner_name not in COMBINERS:
            raise ValueError(f"Combiner desconocido: {self.combiner_name}")
        self.combiner = COMBINERS[self.combiner_name]

        self.output_prefix = self.reward_cfg.get("results_output_prefix", "reward_results")
        self.output_dir = self.reward_cfg.get("results_output_dir", ".")
        self.fail_on_error = bool(self.reward_cfg.get("fail_on_error", True))
        self.last_epoch_results: Optional[pd.DataFrame] = None
        self.last_epoch: Optional[int] = None

        providers_cfg = self.reward_cfg.get("providers", [])
        if not providers_cfg:
            raise ValueError("reward.providers no puede estar vacío")

        self.providers: List[Tuple[str, object]] = []
        used_names = set()
        for entry in providers_cfg:
            if isinstance(entry, str):
                provider_type = entry
                provider_name = entry
                provider_cfg = self.reward_cfg.get(provider_type, {})
            elif isinstance(entry, dict):
                provider_type = entry.get("type")
                if provider_type is None:
                    raise ValueError("Cada provider dict debe incluir la clave 'type'")
                provider_name = entry.get("name", provider_type)
                provider_cfg = {k: v for k, v in entry.items() if k not in {"type", "name"}}
            else:
                raise ValueError("Cada provider debe ser un string o un dict")

            if provider_type not in PROVIDER_MAP:
                raise ValueError(f"Provider desconocido: {provider_type}")

            if provider_name in used_names:
                raise ValueError(f"Nombre de provider repetido: {provider_name}")
            used_names.add(provider_name)

            provider_cls = PROVIDER_MAP[provider_type]
            self.providers.append((provider_name, provider_cls(cfg, provider_cfg)))

    def _normalize_smiles_input(self, molecules: List[str]) -> List[Optional[str]]:
        if self.input_type == "selfies":
            molecules_smiles = []
            for selfie in molecules:
                try:
                    smiles = sf.decoder(selfie)
                    if smiles is None or not isinstance(smiles, str) or len(smiles.strip()) == 0:
                        smiles = None
                    molecules_smiles.append(smiles)
                except Exception:
                    molecules_smiles.append(None)
            return molecules_smiles

        if self.input_type == "smiles":
            normalized = []
            for smi in molecules:
                if smi is None:
                    normalized.append(None)
                    continue
                smi_str = str(smi).strip()
                normalized.append(smi_str if len(smi_str) > 0 else None)
            return normalized

        raise ValueError(f"reward.input_type no soportado: {self.input_type}")

    def _merge_provider_outputs(self, provider_outputs: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
        if len(provider_outputs) == 0:
            raise ValueError("No hay DataFrames de providers para mergear")

        _, first_df = provider_outputs[0]
        merged = first_df.copy()
        for provider_name, df in provider_outputs[1:]:
            overlap_cols = (set(merged.columns) & set(df.columns)) - {"input_idx", "SMILES"}
            if overlap_cols:
                rename_map = {col: f"{provider_name}_{col}" for col in overlap_cols}
                df = df.rename(columns=rename_map)
            merged = merged.merge(df, on=["input_idx", "SMILES"], how="inner")
        return merged

    def _apply_combiner(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.combiner_name == "docking_only":
            docking_col = self.reward_cfg.get("docking_column", "Docking")
            if docking_col not in df.columns:
                raise ValueError(f"Falta columna '{docking_col}'")
            df["Fitness"] = df[docking_col].apply(self.combiner)

        elif self.combiner_name == "docking_multi":
            docking_columns = self.reward_cfg.get("docking_columns")
            if docking_columns is None:
                docking_columns = [c for c in df.columns if c.lower().startswith("docking")]
            if len(docking_columns) == 0:
                raise ValueError("No se encontraron columnas de docking para 'docking_multi'")

            missing_cols = [c for c in docking_columns if c not in df.columns]
            if missing_cols:
                raise ValueError(f"Columnas docking faltantes: {missing_cols}")

            aggregation = self.reward_cfg.get("docking_aggregation", "mean")
            df["Fitness"] = df[docking_columns].apply(
                lambda row: self.combiner(row.to_numpy(dtype=float), aggregation=aggregation),
                axis=1,
            )

        elif self.combiner_name == "docking_logp":
            docking_col = self.reward_cfg.get("docking_column", "Docking")
            logp_col = self.reward_cfg.get("logp_column", "LogP")
            if docking_col not in df.columns or logp_col not in df.columns:
                raise ValueError(f"Faltan columnas '{docking_col}' y/o '{logp_col}'")
            df["Fitness"] = df.apply(
                lambda row: self.combiner(row[docking_col], row[logp_col]),
                axis=1,
            )

        elif self.combiner_name == "mw_only":
            mw_col = self.reward_cfg.get("mw_column", "MW")
            if mw_col not in df.columns:
                raise ValueError(f"Falta columna '{mw_col}'")
            df["Fitness"] = df[mw_col].apply(self.combiner)

        elif self.combiner_name == "weighted_sum":
            weighted_cfg = self.reward_cfg.get("weighted_sum")
            if weighted_cfg is None:
                raise ValueError("Falta reward.weighted_sum para combiner 'weighted_sum'")

            if "columns" in weighted_cfg and "weights" in weighted_cfg:
                columns = weighted_cfg["columns"]
                weights = weighted_cfg["weights"]
                bias = weighted_cfg.get("bias", 0.0)
            else:
                bias = weighted_cfg.get("bias", 0.0)
                columns = [c for c in weighted_cfg.keys() if c != "bias"]
                weights = [weighted_cfg[c] for c in columns]

            missing_cols = [c for c in columns if c not in df.columns]
            if missing_cols:
                raise ValueError(f"Columnas faltantes para weighted_sum: {missing_cols}")

            df["Fitness"] = df[columns].apply(
                lambda row: self.combiner(row.to_numpy(dtype=float), weights, bias=bias),
                axis=1,
            )

        else:
            raise NotImplementedError(f"Combiner '{self.combiner_name}' no implementado")

        return df

    def _save_epoch_results(self, df_temp: pd.DataFrame, epoch: int) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        output_file = os.path.join(self.output_dir, f"{self.output_prefix}_{epoch}.csv")
        df_temp.to_csv(output_file, index=False)

    def __call__(self, molecules: List[str], epoch: int) -> List[float]:
        molecules_smiles = self._normalize_smiles_input(molecules)

        if len(molecules_smiles) == 0:
            return []

        valid_pairs = [
            (idx, smi) for idx, smi in enumerate(molecules_smiles) if smi is not None
        ]

        if len(valid_pairs) == 0:
            return [0.0] * len(molecules)

        valid_indices = [idx for idx, _ in valid_pairs]
        valid_smiles = [smi for _, smi in valid_pairs]

        try:
            provider_outputs = [
                (provider_name, provider.compute(valid_smiles, epoch))
                for provider_name, provider in self.providers
            ]
            df_temp = self._merge_provider_outputs(provider_outputs)
            df_temp = self._apply_combiner(df_temp)
            self._save_epoch_results(df_temp, epoch)
            self.last_epoch_results = df_temp.copy()
            self.last_epoch = int(epoch)

            valid_rewards = (
                df_temp.sort_values("input_idx")["Fitness"]
                .fillna(0.0)
                .astype(float)
                .tolist()
            )

            if len(valid_rewards) != len(valid_smiles):
                raise RuntimeError(
                    f"Longitud de rewards válidas incorrecta: {len(valid_rewards)} vs {len(valid_smiles)}"
                )

            rewards = [0.0] * len(molecules)
            for original_idx, reward in zip(valid_indices, valid_rewards):
                rewards[original_idx] = float(reward)

            return rewards

        except Exception as e:
            print(f"Error durante reward_fn: {e}")
            self.last_epoch_results = None
            self.last_epoch = None
            if self.fail_on_error:
                raise
            return [0.0] * len(molecules)