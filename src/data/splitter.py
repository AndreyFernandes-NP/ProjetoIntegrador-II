"""
splitter.py
-----------
Utilitário para dividir o dataset em treino, validação e teste, com auditoria de sobreposição e preservação de grupos/ordem temporal.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.config import PATHS
from src.data.processor import project_path, require_columns
from src.data.validator import inspect_images, missing, write_json

PARTS = ("train", "validation", "test")
FORMAT_VERSION = 1

@dataclass
class DatasetSplit:
    frames: dict[str, pd.DataFrame]
    assignments: pd.Series  # index: ID do chamado, valor: train/validation/test
    report: dict
    reused: bool

def split_config(config: dict) -> dict:
    defaults = {
        "enabled": True, "strategy": "stratified", "id_column": "ticket_id",
        "train_size": 0.80, "validation_size": 0.10, "test_size": 0.10,
        "seed": 42, "output": "data/processed/splits/chamados.json",
        "group_column": None, "group_mode": "column", "group_candidates": 512,
        "max_size_deviation": 0.10, "time_column": "created_at",
        "require_all_classes": True, "audit_columns": [], "overlap_policy": "warn",
        "image_column": None, "image_base_dir": "data",
    }

    if not isinstance(config, dict) or set(config) - set(defaults):
        raise ValueError("Configuração de split inválida ou com campos desconhecidos")
    
    cfg = {**defaults, **config}
    for key in ["enabled", "require_all_classes"]:
        if not isinstance(cfg[key], bool):
            raise ValueError(f"split.{key} deve ser true/false")
        
    if cfg["strategy"] not in {"stratified", "group", "temporal"}:
        raise ValueError("strategy deve ser stratified, group ou temporal")
    
    sizes = [cfg[f"{part}_size"] for part in PARTS]

    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v < 1 for v in sizes):
        raise ValueError("As três proporções devem estar entre 0 e 1")
    
    if not np.isclose(sum(sizes), 1.0, rtol=0, atol=1e-9):
        raise ValueError("As proporções de treino, validação e teste devem somar 1")
    
    if type(cfg["seed"]) is not int or not 0 <= cfg["seed"] < 2**32:
        raise ValueError("seed deve ser inteiro entre 0 e 2**32-1")
    
    if type(cfg["group_candidates"]) is not int or not 1 <= cfg["group_candidates"] <= 10000:
        raise ValueError("group_candidates deve ser inteiro entre 1 e 10000")
    
    tolerance = cfg["max_size_deviation"]

    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not 0 <= tolerance <= 1:
        raise ValueError("max_size_deviation deve estar entre 0 e 1")
    
    if cfg["group_mode"] not in {"column", "image_hash"}:
        raise ValueError("group_mode deve ser column ou image_hash")
    
    if cfg["overlap_policy"] not in {"warn", "error"}:
        raise ValueError("overlap_policy deve ser warn ou error")
    
    if not isinstance(cfg["audit_columns"], list) or not all(isinstance(c, str) for c in cfg["audit_columns"]):
        raise ValueError("audit_columns deve ser uma lista de colunas")
    
    for key in ["id_column", "time_column", "output", "image_base_dir"]:
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError(f"split.{key} deve ser texto não vazio")
        
    for key in ["group_column", "image_column"]:
        if cfg[key] is not None and (not isinstance(cfg[key], str) or not cfg[key].strip()):
            raise ValueError(f"split.{key} deve ser texto ou null")
        
    return cfg

def fingerprint(df: pd.DataFrame, image_hashes: pd.Series | None = None) -> str:
    """Recebe linhas já ordenadas por ID; inclui valores, schema e bytes das imagens."""

    digest = sha256()
    digest.update(json.dumps([(str(c), str(df[c].dtype)) for c in df.columns]).encode())
    digest.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())

    if image_hashes is not None:
        digest.update(pd.util.hash_pandas_object(image_hashes, index=False).to_numpy().tobytes())

    return digest.hexdigest()

def assign_new(df: pd.DataFrame, target: str, cfg: dict, groups: pd.Series | None) -> np.ndarray:
    sizes = np.array([cfg[f"{part}_size"] for part in PARTS])
    y = df[target].astype(str)

    positions = np.arange(len(df))
    
    if cfg["strategy"] == "stratified":
        try:
            train, rest = train_test_split(positions, train_size=sizes[0], random_state=cfg["seed"], stratify=y)
            val, test = train_test_split(rest, test_size=sizes[2] / sizes[1:].sum(),
                                         random_state=cfg["seed"], stratify=y.iloc[rest])
        except ValueError as exc:
            raise ValueError(f"Estratificação inviável; confira quantidade por classe e proporções: {exc}") from exc
        
    elif cfg["strategy"] == "temporal":
        timestamps = pd.to_datetime(df[cfg["time_column"]], errors="raise", utc=True)
        if timestamps.isna().any():
            raise ValueError("Datas ausentes não podem ser usadas na divisão temporal")
        
        order = np.argsort(timestamps.to_numpy(), kind="stable")
        ordered = timestamps.iloc[order].reset_index(drop=True)

        # Não divide chamados com o mesmo instante entre conjuntos diferentes
        boundaries = np.flatnonzero(ordered.ne(ordered.shift()).to_numpy())[1:]
        if len(boundaries) < 2:
            raise ValueError("Divisão temporal exige pelo menos três instantes distintos")
        
        first = boundaries[:-1][np.argmin(abs(boundaries[:-1] - len(df) * sizes[0]))]
        later = boundaries[boundaries > first]
        second = later[np.argmin(abs(later - len(df) * sizes[:2].sum()))]
        train, val, test = np.split(order, [first, second])
    else:
        if groups is None or groups.nunique() < 3:
            raise ValueError("Divisão por grupos exige pelo menos três grupos distintos")
        
        classes = sorted(y.unique())
        codes = pd.Categorical(y, categories=classes).codes
        global_distribution = np.bincount(codes, minlength=len(classes)) / len(df)
        best, best_score = None, float("inf")
        outer = GroupShuffleSplit(n_splits=cfg["group_candidates"], test_size=sizes[1:].sum(),
                                  random_state=cfg["seed"])
        
        # Candidatos são comparados apenas por tamanhos e distribuição de classes,
        # nenhum resultado de modelo/teste participa da escolha
        for trial, (candidate_train, rest) in enumerate(outer.split(df, groups=groups)):
            inner = GroupShuffleSplit(n_splits=1, test_size=sizes[2] / sizes[1:].sum(),
                                      random_state=(cfg["seed"] + trial + 1) % 2**32)
            try:
                v, t = next(inner.split(df.iloc[rest], groups=groups.iloc[rest]))
                
            except ValueError:
                continue

            candidate = (candidate_train, rest[v], rest[t])
            counts = [np.bincount(codes[idx], minlength=len(classes)) for idx in candidate]

            if cfg["require_all_classes"] and any((count == 0).any() for count in counts):
                continue

            fractions = np.array([len(idx) / len(df) for idx in candidate])

            if np.max(abs(fractions - sizes)) > cfg["max_size_deviation"]:
                continue

            score = float(abs(fractions - sizes).sum()) + sum(float(abs(count / count.sum() - global_distribution).mean()) for count in counts)

            if score < best_score:
                best, best_score = candidate, score

        if best is None:
            raise ValueError("Não foi encontrada divisão por grupos com os limites e classes exigidos. "
                             "Revise os dados, proporções, max_size_deviation ou group_candidates.")
        
        train, val, test = best
        
    assignment = np.empty(len(df), dtype=object)
    for label, indices in zip(PARTS, (train, val, test)):
        assignment[indices] = label

    return assignment

def verify_assignment(df, assignments, target, cfg, groups) -> None:
    if len(assignments) != len(df) or not set(assignments).issubset(PARTS):
        raise ValueError("Manifesto contém conjuntos inválidos")
    
    all_classes = set(df[target].astype(str))
    for part in PARTS:
        subset = df.loc[assignments == part]
        if subset.empty:
            raise ValueError(f"Conjunto vazio: {part}")
        
        if cfg["require_all_classes"] and set(subset[target].astype(str)) != all_classes:
            raise ValueError(f"{part} não contém todas as classes; reveja a estratégia/dados")
        
        if abs(len(subset) / len(df) - cfg[f"{part}_size"]) > cfg["max_size_deviation"]:
            raise ValueError(f"Proporção de {part} fora de max_size_deviation")
        
    if groups is not None:
        check = pd.DataFrame({"group": groups.to_numpy(), "split": assignments})
        if check.groupby("group")["split"].nunique().gt(1).any():
            raise ValueError("Um mesmo grupo aparece em mais de um conjunto")
        
    if cfg["strategy"] == "temporal":
        dates = pd.to_datetime(df[cfg["time_column"]], errors="raise", utc=True)
        if dates.isna().any():
            raise ValueError("Datas ausentes na divisão temporal")
        
        for left, right in zip(PARTS, PARTS[1:]):
            if dates[assignments == left].max() >= dates[assignments == right].min():
                raise ValueError("Manifesto viola a ordem temporal ou divide instantes iguais")

def audit_overlap(df, assignments, cfg, image_hashes):
    audit = {}
    series = {col: df[col] for col in cfg["audit_columns"]}

    if image_hashes is not None:
        series["image_content_sha256"] = image_hashes

    for col, values in series.items():
        sets = {part: set(values[(assignments == part) & ~missing(values)].astype(str)) for part in PARTS}
        audit[col] = {f"{left}__{right}": len(sets[left] & sets[right])
                      for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]}
    return audit

def get_or_create_split(df: pd.DataFrame, models_config: dict, *, root: Path = PATHS.root,
                        dry_run: bool = False, rebuild: bool = False) -> DatasetSplit | None:
    global_cfg = models_config["global_config"]
    raw_cfg = global_cfg.get("split")

    if raw_cfg is None:
        return None
    
    cfg = split_config(raw_cfg)
    if not cfg["enabled"]:
        if rebuild:
            raise ValueError("Não é possível recriar uma divisão desabilitada")
        
        return None
    
    target, id_col = global_cfg.get("target_col"), cfg["id_column"]
    if not isinstance(target, str) or not target:
        raise ValueError("A divisão exige global_config.target_col")
    
    needed = [id_col, target, *cfg["audit_columns"]]
    if cfg["strategy"] == "temporal":
        needed.append(cfg["time_column"])

    if cfg["strategy"] == "group":
        if not cfg["group_column"] or cfg["group_column"] in {id_col, target}:
            raise ValueError("group_column deve ser definido e diferente do ID e do alvo")
        
        needed.append(cfg["group_column"])

    if cfg["image_column"]:
        needed.append(cfg["image_column"])

    require_columns(df, needed, "Split")
    for col in [id_col, target]:
        if missing(df[col]).any():
            raise ValueError(f"Split: '{col}' não pode conter nulos/vazios")
        
    work = df.copy()
    work[id_col] = work[id_col].astype("string")
    if work[id_col].duplicated().any():
        raise ValueError(f"Split: '{id_col}' deve ser único")
    
    work = work.sort_values(id_col, kind="stable").reset_index(drop=True)
    groups, image_hashes = None, None
    image_col = cfg["image_column"]

    if cfg["strategy"] == "group" and cfg["group_mode"] == "image_hash":
        if image_col != cfg["group_column"]:
            raise ValueError("Para image_hash, image_column deve ser igual a group_column")
        
    if image_col:
        valid, invalid = inspect_images(work[image_col], project_path(cfg["image_base_dir"], root))

        if invalid:
            raise ValueError(f"Split: {len(invalid)} imagem(ns) inexistente(s) ou ilegível(is)")
        
        image_hashes = work[image_col].astype("string").map({k: v["sha256"] for k, v in valid.items()})
    if cfg["strategy"] == "group":
        groups = image_hashes if cfg["group_mode"] == "image_hash" else work[cfg["group_column"]].astype("string")
        if groups is None or missing(groups).any():
            raise ValueError("Coluna de agrupamento não pode conter valores ausentes")
        
    dataset_hash = fingerprint(work, image_hashes)
    signature = {"target_col": target, **{k: v for k, v in cfg.items() if k not in {"output", "enabled"}}}
    destination = project_path(cfg["output"], root)

    if destination.suffix.lower() != ".json":
        raise ValueError("split.output deve ser um arquivo .json")
    
    if destination == project_path(global_cfg["dataset"], root):
        raise ValueError("O manifesto não pode substituir o dataset")
    
    saved = {}
    reused = destination.exists() and not rebuild
    if reused:
        try:
            saved = json.loads(destination.read_text(encoding="utf-8"))
            if (saved["format_version"] != FORMAT_VERSION or saved["dataset_sha256"] != dataset_hash
                    or saved["configuration"] != signature):
                raise ValueError("Dataset/configuração mudou. Use --rebuild-split para criar uma nova divisão.")
            
            records = saved["assignments"]
            if not isinstance(records, list) or len(records) != len(work):
                raise ValueError("Quantidade de IDs no manifesto não corresponde ao dataset")
            
            by_id = {item["id"]: item["split"] for item in records}
            if len(by_id) != len(work) or set(by_id) != set(work[id_col]):
                raise ValueError("IDs ausentes, extras ou duplicados no manifesto")
            
            assignments = work[id_col].map(by_id).to_numpy()
            expected_hash = sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
            if saved["assignments_sha256"] != expected_hash:
                raise ValueError("Conteúdo do manifesto foi alterado; restaure ou recrie a divisão")
            
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Manifesto inválido; restaure ou use --rebuild-split") from exc
    else:
        assignments = assign_new(work, target, cfg, groups)

    verify_assignment(work, assignments, target, cfg, groups)
    overlaps = audit_overlap(work, assignments, cfg, image_hashes)
    warnings = [f"'{column}': valores compartilhados {pairs}" for column, pairs in overlaps.items() if any(pairs.values())]

    if warnings and cfg["overlap_policy"] == "error":
        raise ValueError("Sobreposição entre conjuntos: " + "; ".join(warnings))
    
    frames = {part: work.loc[assignments == part].copy() for part in PARTS}
    summary = {part: {"rows": len(frame), "fraction": len(frame) / len(work),
                      "classes": {str(k): int(v) for k, v in frame[target].value_counts().items()},
                      "groups": int(groups[assignments == part].nunique()) if groups is not None else None}
               for part, frame in frames.items()}
    
    records = [{"id": str(value), "split": str(part)} for value, part in zip(work[id_col], assignments)]
    report = {"format_version": FORMAT_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
              "dataset_sha256": dataset_hash, "configuration": signature, "summary": summary,
              "overlap": overlaps, "warnings": warnings, "assignments": records,
              "assignments_sha256": sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()}
    
    if reused:
        report["created_at"] = saved["created_at"]
        
    elif not dry_run:
        write_json(destination, report)

    print(f"[Split / {'Reutilizado' if reused else 'Simulado' if dry_run else 'Criado'}] " +
          ", ".join(f"{part}={len(frames[part])}" for part in PARTS))
    
    for message in warnings:
        print(f"  [warning] {message}")

    return DatasetSplit(frames, pd.Series(assignments, index=work[id_col].tolist(), name="split"), report, reused)
