"""
fusion_pipeline.py
------------------
Busca hiperparâmetros da MLP multimodal usando a divisão já persistida, seleciona
pela validação e avalia somente o vencedor no teste. Aceita configurações legadas.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import ParameterSampler
from threadpoolctl import threadpool_limits
import yaml

from src.config import PATHS
from src.data.processor import load_yaml, project_path, require_columns
from src.data.splitter import PARTS
from src.data.validator import write_json
from src.models.fusion import EarlyFusionClassifier, feature_columns, classification_metrics
from src.models.reporting import RunReport, file_hash

# A esse ponto to tentando tudo e todos pra otimizar e melhorar essa bagaça

def validate_fusion_config(config: dict) -> dict:
    """Valida lista de classes, seleção explícita de features e espaços de busca."""
    cfg = config["multimodal"]

    if config.get("schema_version") != 3 or cfg.get("enabled") is not True:
        raise ValueError("Configure schema_version: 3 e multimodal.enabled: true")
    
    classes = cfg["classes"]
    if not isinstance(classes, list) or len(classes) < 2 or not all(isinstance(v, str) and v.strip() for v in classes) or len(set(classes)) != len(classes):
        raise ValueError("multimodal.classes deve conter classes distintas e não vazias")
    
    f = cfg["features"]
    for columns in [f["numeric"], f["categorical"], *f["text"].values()]:
        if not isinstance(columns, list) or not all(isinstance(c, str) and c for c in columns):
            raise ValueError("Declare listas de nomes nas features multimodais")
        
    for name, columns in f["text"].items():
        if not name.isidentifier() or name in {"numeric", "categorical"} or not columns:
            raise ValueError("Blocos textuais devem ter nome simples e colunas não vazias")
        
    columns = feature_columns(cfg)
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("Features multimodais vazias ou duplicadas")
    
    global_cfg = config["global_config"]
    banned = {global_cfg["target_col"], global_cfg["split"]["id_column"], "split", "customer_id",
              "image_path", "assigned_team", "resolution_time_hours", "status", "priority"}
    
    if banned.intersection(columns):
        raise ValueError(f"Features com risco de vazamento: {sorted(banned.intersection(columns))}")
    
    if "tabular_profile_cluster" in columns and "tabular_profile_cluster" not in f["categorical"]:
        raise ValueError("tabular_profile_cluster deve ser categórico")
    
    if cfg["search"]["metric"] not in {"accuracy", "balanced_accuracy", "f1_macro"}:
        raise ValueError("Métrica de seleção inválida")
    
    for key in ("epochs", "patience", "cpu_threads"):
        value = cfg["training"][key]
        if type(value) is not int or value < 1:
            raise ValueError(f"training.{key} deve ser inteiro positivo")
        
    delta = cfg["training"]["min_delta"]
    if not isinstance(delta, (int, float)) or not np.isfinite(delta) or delta < 0:
        raise ValueError("min_delta deve ser finito e não negativo")
    
    spaces = cfg["search"]["spaces"]
    if not spaces or set(spaces) - {"logistic", "mlp"}:
        raise ValueError("Espaços disponíveis: logistic e mlp")

    if "logistic" in spaces:
        value = cfg["training"].get("logistic_max_iter")
        if type(value) is not int or value < 1:
            raise ValueError("training.logistic_max_iter deve ser positivo quando logistic estiver habilitado")
    
    common = {"text_max_features", "text_min_df", "text_ngram_max", "class_weight"}
    for kind, space in spaces.items():
        if type(space["trials"]) is not int or not 1 <= space["trials"] <= 100:
            raise ValueError("trials deve estar entre 1 e 100 por família")
        
        expected = common | ({"C"} if kind == "logistic" else {"hidden_dims", "dropout", "learning_rate", "weight_decay", "batch_size"})
        params = space["parameters"]
        optional = {"l2_lambda"} if kind == "mlp" else set()
        if not expected.issubset(params) or set(params) - expected - optional or any(not isinstance(v, list) or not v for v in params.values()):
            raise ValueError(f"Parâmetros incompatíveis em {kind}: {sorted(expected)}")
        
        for key, values in params.items():
            for value in values:
                if key == "class_weight":
                    valid = value in (None, "balanced")

                elif key == "hidden_dims":
                    valid = isinstance(value, list) and bool(value) and all(type(v) is int and v > 0 for v in value)

                elif key in {"text_max_features", "text_min_df", "text_ngram_max", "batch_size"}:
                    valid = type(value) is int and value > 0

                else:
                    valid = not isinstance(value, bool) and isinstance(value, (int, float)) and np.isfinite(value)

                    if valid:
                        valid = 0 <= value < 1 if key == "dropout" else value >= 0 if key in {"weight_decay", "l2_lambda"} else value > 0

                if not valid:
                    raise ValueError(f"Valor inválido em {kind}.{key}: {value}")
    return cfg

def sample_candidates(config: dict, seed: int) -> list[dict]:
    """Amostra combinações sem reposição por família, com orçamento e seed fixos."""
    candidates = []

    for i, (kind, space) in enumerate(config["search"]["spaces"].items()):
        total = int(np.prod([len(v) for v in space["parameters"].values()]))

        for params in ParameterSampler(space["parameters"], n_iter=min(space["trials"], total), random_state=seed + i):
            candidates.append({"kind": kind, **params})

    return candidates

def read_feature_source(config: dict, *, from_run=None, input_path=None, root: Path = PATHS.root):
    """Lê uma execução concluída ou Parquet explícito."""
    if (from_run is None) == (input_path is None):
        raise ValueError("Informe exatamente um de from_run ou input_path")
    
    parent = None
    if from_run is not None:
        parent = project_path(from_run, root)
        report = json.loads((parent / "run.json").read_text(encoding="utf-8"))

        if report.get("status") != "completed" or report.get("format_version") != 3:
            raise ValueError("A origem deve ser uma execução concluída de features schema 3")
        
        for name in ("enriched.parquet", "models.yaml", "feature_schema.json", "split.json"):
            if file_hash(parent / name) != report["artifacts_sha256"][name]:
                raise ValueError(f"Origem alterada: {name}")
            
        source_config = load_yaml(parent / "models.yaml")
        for current, source in [(config["global_config"]["target_col"], source_config["global_config"]["target_col"]),
                                (config["global_config"]["split"]["id_column"], source_config["global_config"]["split"]["id_column"])]:
            if current != source:
                raise ValueError("ID/target da configuração diferem da execução de origem")
            
        path = parent / "enriched.parquet"
    else:
        assert input_path is not None
        path = project_path(input_path, root)

    data = pd.read_parquet(path)
    id_col, target = config["global_config"]["split"]["id_column"], config["global_config"]["target_col"]

    require_columns(data, [id_col, target, "split", *feature_columns(config["multimodal"])], "Dataset enriquecido")
    if data.columns.duplicated().any() or data[id_col].isna().any() or data[id_col].astype(str).str.strip().eq("").any() or data[id_col].astype(str).duplicated().any():
        raise ValueError("IDs/colunas inválidos ou duplicados no dataset enriquecido")
    
    if data["split"].isna().any() or set(data["split"].unique()) != set(PARTS):
        raise ValueError("A origem deve conter somente train, validation e test, todos não vazios")
    
    if data[target].isna().any() or not set(data[target].astype(str)).issubset(config["multimodal"]["classes"]):
        raise ValueError("Target ausente ou fora da lista de classes")
    
    for part in PARTS:
        if set(data.loc[data["split"] == part, target].astype(str)) != set(config["multimodal"]["classes"]):
            raise ValueError(f"{part} deve conter todas as classes para avaliação comparável")
        
    return data, path, parent

def text_overlap(data: pd.DataFrame, columns: list[str]) -> dict:
    """Audita textos iguais entre partições."""
    keys = data[columns].astype("string").fillna("").agg("\n".join, axis=1).str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()
    sets = {p: set(keys[data["split"] == p]) - {""} for p in PARTS}

    return {f"{a}__{b}": len(sets[a] & sets[b]) for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]}

def evaluate_partition(model, frame: pd.DataFrame, target: str, id_col: str, directory: Path, name: str) -> dict:
    """Grava probabilidades, classificação por classe e matriz de confusão de uma partição."""
    
    start = perf_counter()
    probabilities = model.predict_proba(frame)
    y = model.encode_labels(frame[target])
    predicted = model.classes[probabilities.argmax(axis=1)]

    result = classification_metrics(y, probabilities, len(model.classes))
    result["inference_seconds"] = perf_counter() - start
    result["classification_report"] = classification_report(frame[target].astype(str), predicted,
        labels=model.classes.tolist(), output_dict=True, zero_division=0)
    
    out = pd.DataFrame({id_col: frame[id_col].to_numpy(), "y_true": frame[target].to_numpy(),
                        "predicted_target": predicted, "prediction_score": probabilities.max(axis=1)})
    
    for i, label in enumerate(model.classes):
        out[f"probability__{label}"] = probabilities[:, i]

    out.to_parquet(directory / f"{name}_predictions.parquet", index=False)
    confusion = pd.DataFrame(confusion_matrix(frame[target].astype(str), predicted, labels=model.classes),
                             index=model.classes, columns=model.classes)
    
    confusion.index.name = "real / previsto"
    confusion.to_csv(directory / f"{name}_confusion.csv", encoding="utf-8-sig")

    return result

def train_fusion(config_path=None, *, from_run=None, input_path=None, dry_run=False, root: Path = PATHS.root):
    """Seleciona hiperparâmetros na validação, sem nenhum outro refit depois.

    Uma busca aleatória com holdout é utilizada, preservando os grupos da divisão.
    Não usa CV aleatória por linha e não inclui teste em qualquer chamada de fit.
    """
    import torch
    root = Path(root)

    config = load_yaml(project_path(config_path or "src/config/models.yaml", root))
    cfg = validate_fusion_config(config)
    data, source, parent = read_feature_source(config, from_run=from_run, input_path=input_path, root=root)
    seed = config["global_config"]["split"]["seed"]
    candidates = sample_candidates(cfg, seed)
    parts = {p: data.loc[data["split"] == p].copy() for p in PARTS}

    if dry_run:
        print(f"[Fusão / dry-run] {len(candidates)} tentativas; " + ", ".join(f"{p}={len(f)}" for p, f in parts.items()))
        return None

    report = RunReport(project_path(cfg.get("reports_directory", "reports/fusion"), root), config, root)
    directory = report.directory
    artifacts = root / "models" / "fusion" / directory.name
    id_col, target = config["global_config"]["split"]["id_column"], config["global_config"]["target_col"]
    parent_ref = str(parent.relative_to(root)) if parent is not None and parent.is_relative_to(root) else str(parent) if parent else None
    report.metadata.update(format_version=4, stage="fusion", seed=seed, input_sha256=file_hash(source),
        input_file=str(source), features_run=parent_ref, feature_provenance="verified_run" if parent else "parquet_only",
        model_directory=str(artifacts.relative_to(root)), selection_metric=cfg["search"]["metric"],
        partition_rows={p: len(f) for p, f in parts.items()})
    
    previous_threads = torch.get_num_threads()
    trials, best_model, best_key = [], None, None
    try:
        torch.set_num_threads(cfg["training"]["cpu_threads"])
        report.metadata["packages"]["torch"] = version("torch")

        if parent:
            shutil.copyfile(parent / "models.yaml", directory / "features_models.yaml")
            shutil.copyfile(parent / "split.json", directory / "split.json")

        data[[id_col, "split"]].to_csv(directory / "assignments.csv", index=False)
        report.metadata["text_overlap"] = {name: text_overlap(data, columns) for name, columns in cfg["features"]["text"].items()}

        report.flush()
        with threadpool_limits(limits=cfg["training"]["cpu_threads"]):
            for number, params in enumerate(candidates, 1):
                start = perf_counter()
                trial = {"trial": number, "params": params}
                model = EarlyFusionClassifier(cfg, params, seed)

                try:
                    model.fit(parts["train"], parts["train"][target], parts["validation"], parts["validation"][target])

                    probabilities = model.predict_proba(parts["validation"])
                    metrics = classification_metrics(model.encode_labels(parts["validation"][target]), probabilities, len(model.classes))
                    trial.update(status="completed", validation=metrics, details=model.details, input_dimensions=model.input_dimensions)
                    key = (metrics[cfg["search"]["metric"]], metrics["balanced_accuracy"], -metrics["log_loss"])

                    if best_key is None or key > best_key:
                        best_key, best_model = key, model
                        report.metadata["best_trial"] = number

                    print(f"[Busca {number}/{len(candidates)}] {params['kind']} validação {cfg['search']['metric']}={key[0]:.4f}", flush=True)

                except (ValueError, RuntimeError) as exc:
                    trial.update(status="failed", error=f"{type(exc).__name__}: {exc}")

                    print(f"[Busca {number}/{len(candidates)}] falhou: {exc}", flush=True)

                trial["seconds"] = perf_counter() - start
                trials.append(trial)
                (directory / "trials.json").write_text(
                    json.dumps(trials, ensure_ascii=False, indent=2), encoding="utf-8"
                )

                del model
                
        if best_model is None:
            raise ValueError("Todas as tentativas falharam; consulte trials.json")
        
        best_model.save(artifacts)
        (directory / "best_params.yaml").write_text(yaml.safe_dump(best_model.params, sort_keys=False), encoding="utf-8")
        report.metadata.update(selection_complete=True, input_dimensions=best_model.input_dimensions,
                               best_params=best_model.params)
        
        report.flush()
        metrics = {}

        for part in PARTS:
            metrics[part] = evaluate_partition(best_model, parts[part], target, id_col, directory, part)

        write_json(directory / "metrics.json", metrics)
        pd.DataFrame([{ "trial": t["trial"], "kind": t["params"]["kind"], "status": t["status"],
            "seconds": t["seconds"], **t.get("validation", {}), "params": json.dumps(t["params"])} for t in trials]).to_csv(directory / "trials.csv", index=False)
        
        report.metadata["model_artifacts_sha256"] = {str(p.relative_to(artifacts)): file_hash(p) for p in artifacts.rglob("*") if p.is_file()}
        report.finish()

        print(f"[Fusão] Teste: accuracy={metrics['test']['accuracy']:.4f}; F1 macro={metrics['test']['f1_macro']:.4f}")
        print(f"[Fusão] Relatório: {directory}")

        return directory
    
    except Exception as exc:
        report.finish(error=exc)
        raise

    finally:
        torch.set_num_threads(previous_threads)

def verify_fusion_run(run_directory: Path, root: Path = PATHS.root) -> tuple[dict, Path]:
    """Verifica hashes de uma execução concluída antes de restaurar artefatos locais."""
    directory = project_path(run_directory, root)
    report = json.loads((directory / "run.json").read_text(encoding="utf-8"))

    if report.get("status") != "completed" or report.get("format_version") != 4:
        raise ValueError("Informe uma execução multimodal concluída, formato 4")
    
    artifacts = project_path(report["model_directory"], root)
    if not artifacts.is_relative_to((Path(root) / "models" / "fusion").resolve()):
        raise ValueError("Diretório de artefatos inválido")
    
    for base, hashes in [(directory, report["artifacts_sha256"]), (artifacts, report["model_artifacts_sha256"])]:
        for relative, expected in hashes.items():
            path = (base / relative).resolve()

            if not path.is_relative_to(base.resolve()) or file_hash(path) != expected:
                raise ValueError(f"Artefato alterado: {relative}")
            
    return report, artifacts

def main() -> None:
    """Treina a fusão usando um dataset já enriquecido, com modo dry-run sem gravação."""
    parser = argparse.ArgumentParser(description="Busca e treino multimodal sobre features existentes")
    parser.add_argument("--config", default="src/config/models.yaml")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-run", help="Pasta reports/runs/... de features concluídas")
    source.add_argument("--input", help="Parquet enriquecido com split; sem restauração automática dos geradores")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        train_fusion(args.config, from_run=args.from_run, input_path=args.input, dry_run=args.dry_run)

    except (ValueError, KeyError, OSError, ImportError) as exc:
        parser.exit(1, f"[Erro] {exc}\n")

if __name__ == "__main__":
    main()
