"""
features_pipeline.py
--------------------
Executa visão, PLN e perfis tabulares em sequência, valida suas saídas e salva um
dataset enriquecido final.
"""
import argparse
import json
from importlib.metadata import version
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import pandas as pd

from src.config import PATHS
from src.data.processor import load_yaml, project_path, require_columns
from src.data.splitter import get_or_create_split, PARTS, split_config
from src.data.validator import write_json
from src.models.feature_common import download_pretrained, hf_source
from src.models.language import ContextualNLP, SemanticNLP, validate_topic_settings, validate_semantic_settings
from src.models.profiles import TabularProfiles
from src.models.vision import ImageCaptioner, caption_partitions
from src.models.reporting import RunReport, file_hash

ADAPTERS = {"blip_captioning": ImageCaptioner, "contextual_nli": ContextualNLP,
            "semantic_nlp": SemanticNLP, "tabular_profiles": TabularProfiles}
LANGUAGE_ADAPTERS = {"contextual_nli", "semantic_nlp"}
CONTRACTS = {
    "blip_captioning": {"image_description": ("string", "add"), "image_score": ("float", "add")},
    "contextual_nli": {"sentiment_score": ("float", "replace"), "main_topic": ("string", "add"),
                       "semantic_margin": ("float", "add"), "urgency_score": ("float", "add")},
    "tabular_profiles": {"tabular_profile_cluster": ("category", "add")},
}

def output_contract(cfg: dict) -> dict[str, tuple[str, str]]:
    """Deriva o contrato semântico das colunas configuradas e mantém os legados."""
    if cfg["adapter"] == "semantic_nlp":
        validate_semantic_settings(cfg)

        return {"sentiment_score": ("float", "replace"),
                **{name: ("float", "add") for name in cfg["semantic"]["prototypes"]}}
    
    return CONTRACTS[cfg["adapter"]]

def added_column_count(config: dict) -> int:
    """Conta novas colunas pelos contratos; substituições não ampliam o dataset."""

    return sum(spec["action"] == "add" for name in config["enrichment"]["stages"] for spec in config["modelos"][name]["outputs"])

def positive_integer(value, field: str) -> None:
    """Valida parâmetros de tamanho/treino antes de baixar os modelos."""
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} deve ser um inteiro positivo")

def validate_features_config(config: dict) -> list[str]:
    """Valida dependências, contratos de saída e ausência de target nas entradas."""
    if config.get("schema_version") != 3 or type(config.get("multimodal", {}).get("enabled")) is not bool:
        raise ValueError("Esta etapa exige schema_version: 3 e multimodal.enabled booleano")
    
    global_cfg = config["global_config"]
    split = split_config(global_cfg["split"])

    if not split["enabled"]:
        raise ValueError("A geração de colunas exige divisão habilitada")
    
    forbidden = {global_cfg["target_col"], split["id_column"], "split"}
    if len(forbidden) != 3:
        raise ValueError("Target, ID e split devem ser distintos")
    
    names = config["enrichment"]["stages"]
    if not isinstance(names, list) or len(names) != 3 or len(set(names)) != 3:
        raise ValueError("Configure exatamente três etapas distintas")
    
    allowed = config["enrichment"]["allow_overwrite"]
    if allowed != ["sentiment_score"]:
        raise ValueError("A única substituição autorizada nesta versão é sentiment_score")
    
    seen, generated, adapters = [], [], []
    for name in names:
        if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
            raise ValueError("Nomes de etapas devem ser identificadores simples")
        
        cfg = config["modelos"][name]
        adapter = cfg["adapter"]
        
        if cfg.get("enabled") is not True or cfg.get("role") != "feature_generator" or adapter not in ADAPTERS:
            raise ValueError(f"Etapa {name} deve ser um gerador habilitado com adaptador conhecido")
        
        features = cfg["features"]
        if not isinstance(features, list) or not features or len(set(features)) != len(features) or forbidden.intersection(features) or cfg.get("target_col") is not None:
            raise ValueError(f"Features inválidas ou vazamento de target/ID/split em {name}")
        
        if not set(cfg.get("depends_on", [])).issubset(seen):
            raise ValueError(f"Dependência fora de ordem em {name}")
        
        specs = cfg["outputs"]
        declared = {s["name"]: (s["dtype"], s["action"]) for s in specs}

        expected = output_contract(cfg)
        if len(declared) != len(specs) or declared != expected or forbidden.intersection(declared):
            raise ValueError(f"Outputs incompatíveis com {adapter}: {expected}")
        
        generated.extend(declared)
        adapters.append(adapter)

        if adapter in {"blip_captioning", *LANGUAGE_ADAPTERS}:
            hf_source(cfg["fonte"])

        if adapter == "semantic_nlp":
            if any(spec.get("range") != [-1, 1] for spec in specs):
                raise ValueError("Sentimento e similaridades semânticas exigem range: [-1, 1]")

        if adapter == "blip_captioning":
            if len(features) != 1:
                raise ValueError("BLIP usa somente a coluna do caminho da imagem")
            
            if type(cfg["training"]["enabled"]) is not bool:
                raise ValueError("training.enabled deve ser true/false")
            
            for field in ("num_beams", "max_new_tokens"):
                positive_integer(cfg["generation"][field], field)

            for field in ("unfreeze_last_layers", "max_length"):
                positive_integer(cfg["training"][field], field)

            if cfg["training"]["max_length"] < 4:
                raise ValueError("max_length visual deve ser >= 4")
            
        if adapter == "contextual_nli":
            validate_topic_settings(cfg["topics"])

            if "sentiment_score" in features:
                raise ValueError("PLN deve recalcular sentimento a partir do texto, sem usar o score antigo")
            
            positive_integer(cfg["batch_size"], "batch_size")

            if cfg["max_length"] < 16 or cfg["urgency"]["context_characters"] < 1:
                raise ValueError("Limites de texto inválidos")
            
            for task in ("sentiment", "topics", "urgency"):
                candidates = cfg[task]["candidates"]
                hypotheses = [c["hypothesis"] for c in candidates]

                if len(candidates) < 2 or len(set(hypotheses)) != len(candidates) or not all(isinstance(h, str) and h.strip() for h in hypotheses):
                    raise ValueError(f"Candidatos inválidos em {task}")
                
                if task == "topics":
                    labels = [c["label"] for c in candidates]

                    if len(set(labels)) != len(labels) or not all(isinstance(v, str) and v.strip() for v in labels):
                        raise ValueError("Labels de tópicos devem ser distintos e não vazios")
                else:
                    values = np.asarray([c["value"] for c in candidates], dtype=float)
                    lo, hi = (-1, 1) if task == "sentiment" else (0, 1)

                    if not np.isfinite(values).all() or (values < lo).any() or (values > hi).any():
                        raise ValueError(f"Valores de {task} fora de [{lo}, {hi}]")
                    
            low, high = cfg["urgency"]["range"]
            if not np.isfinite([low, high]).all() or low >= high:
                raise ValueError("urgency.range exige mínimo < máximo, ambos finitos")
            
            urgency_output = next(s for s in specs if s["name"] == "urgency_score")
            if urgency_output.get("range") != [low, high]:
                raise ValueError("range de urgency_score deve coincidir com urgency.range")
            
        if adapter == "tabular_profiles":
            language_names = [n for n in seen if config["modelos"][n]["adapter"] in LANGUAGE_ADAPTERS]

            if not language_names or language_names[0] not in cfg.get("depends_on", []):
                raise ValueError("Perfis tabulares devem depender explicitamente da etapa PLN")
            
            columns = cfg["preprocessing"]["numeric"] + cfg["preprocessing"]["categorical"]
            if len(columns) != len(set(columns)) or set(columns) != set(features):
                raise ValueError("Preprocessing tabular deve cobrir exatamente as features")
            
            positive_integer(cfg["clustering"]["n_clusters"], "n_clusters")
            for field in ("hidden_dim", "latent_dim"):
                positive_integer(cfg["network"][field], field)

        if "training" in cfg:
            for field in ("epochs", "batch_size", "patience"):
                positive_integer(cfg["training"][field], field)

            rate = float(cfg["training"]["learning_rate"])
            if not np.isfinite(rate) or rate <= 0:
                raise ValueError("learning_rate deve ser positivo e finito")
            
        seen.append(name)

    if (adapters.count("blip_captioning") != 1 or adapters.count("tabular_profiles") != 1
            or sum(a in LANGUAGE_ADAPTERS for a in adapters) != 1
            or len(generated) != len(set(generated))):
        
        raise ValueError("Configure um adaptador de cada modalidade, sem outputs duplicados")
    
    limit = config["enrichment"]["max_new_columns"]
    positive_integer(limit, "max_new_columns")

    if added_column_count(config) > limit:
        raise ValueError("Número de novas colunas excede max_new_columns")
    
    return names

def merge_outputs(frame: pd.DataFrame, outputs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Valida nomes, alinhamento, tipos e limites antes de inserir/substituir colunas."""
    specs = cfg["outputs"]

    if list(outputs.columns) != [s["name"] for s in specs] or not outputs.index.equals(frame.index):
        raise ValueError("Outputs têm colunas, índice ou ordem incompatíveis")
    
    result = frame.copy()
    for spec in specs:
        name, dtype = spec["name"], spec["dtype"]

        if (spec["action"] == "add" and name in frame) or (spec["action"] == "replace" and name not in frame):
            raise ValueError(f"Ação {spec['action']} incompatível com a coluna {name}")
        
        series = outputs[name]
        if series.isna().any():
            raise ValueError(f"Output {name} contém nulos")
        
        if dtype == "string":
            if not series.map(lambda v: isinstance(v, str) and bool(v.strip())).all():
                raise ValueError(f"Output {name} deve conter textos não vazios")
            
            series = series.astype("string")

        elif dtype == "float":
            series = pd.to_numeric(series, errors="raise").astype(float)

            if not np.isfinite(series).all():
                raise ValueError(f"Output {name} não é finito")
            
            low, high = spec.get("range", [-np.inf, np.inf])
            if not series.between(low - 1e-7, high + 1e-7).all():
                raise ValueError(f"Output {name} fora do intervalo [{low}, {high}]")
            
        else:
            values = np.asarray(series, dtype=float)
            count = cfg["clustering"]["n_clusters"]

            if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all() or (values < 0).any() or (values >= count).any():
                raise ValueError("IDs de cluster inválidos")
            
            series = pd.Series(pd.Categorical(values.astype(int), categories=range(count)), index=frame.index)

        result[name] = series

    return result

def download_feature_models(config: dict, root: Path = PATHS.root) -> dict[str, Path]:
    """Baixa checkpoints habilitados, incluindo o encoder semântico; não treina."""
    paths = {}

    for name in validate_features_config(config):
        cfg = config["modelos"][name]

        if "fonte" in cfg:
            paths[name] = download_pretrained(cfg["fonte"], root)

        if cfg.get("adapter") == "contextual_nli" and "fonte" in cfg["topics"]:
            paths[f"{name}.topics"] = download_pretrained(cfg["topics"]["fonte"], root)

        if cfg.get("adapter") == "semantic_nlp":
            paths[f"{name}.semantic"] = download_pretrained(cfg["semantic"]["fonte"], root)
            
    return paths

def run_features(config_path=None, *, df=None, split_result=None, dry_run=False, rebuild_split=False, sources_path=None, root: Path = PATHS.root):
    """Gera as saídas configuradas com ajuste só no treino e relatório por execução.

    Retorna o diretório do relatório, ou None em dry-run. A saída enriquecida é
    reports/runs/<execução>/enriched.parquet, os pesos ficam em models/runs/<execução>.
    """
    root = Path(root)

    config = load_yaml(project_path(config_path or "src/config/models.yaml", root))
    names = validate_features_config(config)
    global_cfg = config["global_config"]
    id_col = global_cfg["split"]["id_column"]

    if df is None:
        df = pd.read_parquet(project_path(global_cfg["dataset"], root))

    if "split" in df:
        raise ValueError("Use o dataset processed, sem a coluna split")
    
    available = set(df.columns)
    for name in names:
        cfg = config["modelos"][name]

        if not set(cfg["features"]).issubset(available):
            raise ValueError(f"Features ausentes para {name}: {set(cfg['features']) - available}")
        
        for spec in cfg["outputs"]:
            exists = spec["name"] in available

            if exists != (spec["action"] == "replace"):
                raise ValueError(f"Colisão/coluna ausente: {spec['name']} ({spec['action']})")
            
            available.add(spec["name"])

    split_result = split_result or get_or_create_split(df, config, root=root, dry_run=dry_run, rebuild=rebuild_split)
    if split_result is None:
        raise ValueError("Divisão de dados ausente")
    
    frames = {p: split_result.frames[p].set_index(id_col, drop=False).copy() for p in PARTS}
    # Um split fornecido externamente precisa corresponder exatamente aos dados.
    combined = pd.concat(list(frames.values())).sort_index()
    original = df.copy()
    original[id_col] = original[id_col].astype("string")
    original = original.set_index(id_col, drop=False)

    if original.index.has_duplicates or combined.index.has_duplicates or not combined.equals(original.sort_index()):
        raise ValueError("Split fornecido não corresponde ao dataset de entrada")
    
    for name in names:
        cfg = config["modelos"][name]

        if cfg["adapter"] == "blip_captioning":
            counts = {p: len(t) for p, t in caption_partitions(cfg, frames, root).items()}
            print(f"[Visão] Imagens únicas por partição: {counts}")
    
    if dry_run:
        print(f"[Features / dry-run] {added_column_count(config)} novas colunas; "
              "substituição de sentiment_score; apenas geração de features")
        
        return None

    report = RunReport(project_path(config.get("reports", {}).get("directory", "reports/runs"), root), config, root)
    directory = report.directory
    artifact_root = root / "models" / "runs" / directory.name
    artifact_root.mkdir(parents=True, exist_ok=False)
    report.metadata.update(format_version=3, stage="features", seed=global_cfg["split"]["seed"],
                           model_directory=str(artifact_root.relative_to(root)), target_used_for_training=False)
    try:
        for package in ("torch", "transformers", "huggingface_hub"):
            report.metadata["packages"][package] = version(package)

        source_cfg = project_path(sources_path or "src/config/sources.yaml", root)
        if source_cfg.exists():
            shutil.copyfile(source_cfg, directory / "sources.yaml")

        write_json(directory / "split.json", split_result.report)
        audit = combined[[id_col, "sentiment_score"]].rename(columns={"sentiment_score": "sentiment_score_original"})
        stats = {}

        for name in names:
            cfg = config["modelos"][name]

            if cfg["adapter"] == "blip_captioning" and cfg["training"]["enabled"]:
                shutil.copyfile(project_path(cfg["training"]["captions_file"], root), directory / "image_captions.csv")

            model = ADAPTERS[cfg["adapter"]](cfg, root, global_cfg["split"]["seed"])
            start = perf_counter()
            model.fit(frames["train"][cfg["features"]], frames["validation"][cfg["features"]])
            fit_seconds = perf_counter() - start

            for part in PARTS:
                outputs = model.transform(frames[part][cfg["features"]])
                frames[part] = merge_outputs(frames[part], outputs, cfg)

                if hasattr(model, "topic_audit"):
                    model.topic_audit.to_parquet(directory / f"{name}_topics_{part}.parquet", index=True)

            model.save(artifact_root / name)
            stats[name] = {**model.details, "fit_seconds": fit_seconds, "total_seconds": perf_counter() - start}

            print(f"[Features] {name}: {[s['name'] for s in cfg['outputs']]}")
            del model  # Libera a rede antes de carregar a próxima modalidade.
            
        enriched = pd.concat([frame.assign(split=part) for part, frame in frames.items()])
        enriched = enriched.loc[original.index]
        audit["sentiment_score"] = enriched["sentiment_score"]
        audit.to_parquet(directory / "sentiment_audit.parquet", index=False)
        schema = {"id_column": id_col, "target_column": global_cfg["target_col"], "stages": names,
                  "outputs": {n: config["modelos"][n]["outputs"] for n in names},
                  "categorical_columns": {"tabular_profile_cluster": list(range(next(config["modelos"][n]["clustering"]["n_clusters"] for n in names if config["modelos"][n]["adapter"] == "tabular_profiles")))}}
        
        write_json(directory / "feature_schema.json", schema)
        write_json(directory / "stage_metrics.json", stats)

        temporary = directory / "enriched.parquet.tmp"
        enriched.to_parquet(temporary, index=False)
        temporary.replace(directory / "enriched.parquet")

        report.metadata.update(rows=len(enriched), added_columns=added_column_count(config), replaced_columns=["sentiment_score"],
            model_artifacts_sha256={str(p.relative_to(artifact_root)): file_hash(p) for p in artifact_root.rglob("*") if p.is_file()})
        report.finish()

        print(f"[Features] Dataset: {directory / 'enriched.parquet'}")

        return directory
    
    except Exception as exc:
        report.finish(error=exc)
        raise

def read_enriched(run_directory: Path) -> pd.DataFrame:
    """Lê a saída concluída e recupera categorias numéricas descritas no schema.

    Alguns leitores Parquet retornam categorias numéricas como int64, por isso essa função
    restaura o dtype category e verifica o hash do dataset antes de carregá-lo.
    """
    directory = Path(run_directory)
    report = json.loads((directory / "run.json").read_text(encoding="utf-8"))

    if report.get("status") != "completed" or report.get("format_version") != 3:
        raise ValueError("O dataset exige uma execução schema 3 concluída")
    
    for name in ("enriched.parquet", "feature_schema.json"):
        if file_hash(directory / name) != report["artifacts_sha256"][name]:
            raise ValueError(f"Artefato alterado: {name}")
        
    data = pd.read_parquet(directory / "enriched.parquet")
    schema = json.loads((directory / "feature_schema.json").read_text(encoding="utf-8"))

    for column, categories in schema["categorical_columns"].items():
        if not data[column].isin(categories).all():
            raise ValueError(f"Categoria inválida em {column}")
        
        data[column] = pd.Categorical(data[column], categories=categories)

    return data

def enrich_new(data: pd.DataFrame, run_directory: Path, *, root: Path = PATHS.root) -> pd.DataFrame:
    """Gera as mesmas colunas com artefatos confiáveis locais.

    Não altera o DataFrame recebido. Se sentiment_score não existir em chamados
    novos, cria um placeholder antes da substituição.
    """
    directory = project_path(run_directory, root)
    report = json.loads((directory / "run.json").read_text(encoding="utf-8"))

    if report.get("status") != "completed" or report.get("format_version") != 3:
        raise ValueError("Informe uma execução de features schema 3 concluída")
    
    artifacts = project_path(report["model_directory"], root)
    if not artifacts.is_relative_to((Path(root) / "models" / "runs").resolve()):
        raise ValueError("Diretório de modelos inválido")
    
    for base, manifest in [(directory, report["artifacts_sha256"]), (artifacts, report["model_artifacts_sha256"])]:
        for relative, expected in manifest.items():
            path = (base / relative).resolve()

            if not path.is_relative_to(base.resolve()) or file_hash(path) != expected:
                raise ValueError(f"Artefato alterado: {relative}")
            
    config = load_yaml(directory / "models.yaml")
    result = data.copy()

    if not result.index.is_unique or result.empty:
        raise ValueError("Use um índice único e dados não vazios")
    
    if "sentiment_score" not in result:
        result["sentiment_score"] = np.nan

    for name in validate_features_config(config):
        cfg = config["modelos"][name]

        require_columns(result, cfg["features"], name)
        model = ADAPTERS[cfg["adapter"]].restore(artifacts / name, cfg, Path(root), config["global_config"]["split"]["seed"])
        result = merge_outputs(result, model.transform(result[cfg["features"]]), cfg)

        del model

    return result

def main() -> None:
    """Executa geração, simulação ou somente download a partir do YAML."""
    parser = argparse.ArgumentParser(description="Geração de colunas antes do multimodal")
    parser.add_argument("--config", default="src/config/models.yaml")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--download-only", action="store_true")
    parser.add_argument("--rebuild-split", action="store_true")
    args = parser.parse_args()
    try:
        if args.download_only:
            for name, path in download_feature_models(load_yaml(project_path(args.config))).items():
                print(f"[Download] {name}: {path}")
        else:
            run_features(args.config, dry_run=args.dry_run, rebuild_split=args.rebuild_split)

    except (ValueError, KeyError, OSError, ImportError) as exc:
        parser.exit(1, f"[Erro] {exc}\n")

if __name__ == "__main__":
    main()
