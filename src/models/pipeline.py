"""
pipeline.py
-----------
Encaminha a configuração atual para geração de colunas sem target e mantém a
compatibilidade com experimentos anteriores de extratores e fusão multimodal.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
from time import perf_counter

import pandas as pd

from src.config import PATHS
from src.data.processor import load_yaml, project_path, require_columns
from src.data.splitter import get_or_create_split, PARTS
from src.data.validator import write_json
from src.models.encoders import create_encoder, save_encoder
from src.models.loader import prepare_models
from src.models.multimodal import FusionClassifier, feature_table, fuse_tables
from src.models.reporting import RunReport, classification_results, file_hash
from src.models.outputs import validate_outputs, project_outputs, output_types

def validate_architecture(config):
    global_cfg = config["global_config"]
    names = config["multimodal"]["encoders"]

    if not isinstance(names, list) or len(names) != 3 or len(set(names)) != 3:
        raise ValueError("multimodal.encoders deve referenciar exatamente três extratores distintos")
    
    modalities, generated = [], []
    forbidden = {global_cfg["target_col"], global_cfg["split"]["id_column"], "split"}

    for name in names:
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError("Use nomes de extratores com letras minúsculas, números e underscore")
        
        model = config["modelos"][name]
        if model.get("enabled") is not True or model.get("role") != "encoder":
            raise ValueError(f"'{name}' deve estar habilitado e ter role: encoder")
        
        if model.get("target_col") is not None:
            raise ValueError("Extratores não devem declarar target_col")
        
        forbidden = {global_cfg["target_col"], global_cfg["split"]["id_column"], "split"}
        if forbidden.intersection(model["features"]):
            raise ValueError("Target, ID e indicador de split não podem entrar nos extratores")
        
        modalities.append(model.get("modality"))
        generated.extend(validate_outputs(model))

    if not all(isinstance(value, str) for value in modalities) or sorted(modalities) != ["tabular", "text", "vision"]:
        raise ValueError("Configure um extrator tabular, um text e um vision")
    
    if global_cfg["split"].get("enabled") is not True:
        raise ValueError("A pipeline multimodal exige divisão habilitada")
    
    if len(generated) != len(set(generated)) or forbidden.intersection(generated):
        raise ValueError("Outputs devem ter nomes únicos globais, distintos de ID/target/split")
    
    limit = config["multimodal"].get("max_generated_columns")
    if type(limit) is not int or limit < 1 or len(generated) > limit:
        raise ValueError("Número de outputs excede max_generated_columns, ou limite inválido")
    
    features = config["multimodal"].get("features")
    if not isinstance(features, list) or not features or not all(isinstance(c, str) for c in features) or len(set(features)) != len(features) or forbidden.intersection(features):
        raise ValueError("Declare multimodal.features sem repetições, ID, target ou split")
    
    categories = config["multimodal"].get("categorical_features", [])
    if not isinstance(categories, list) or not all(isinstance(c, str) for c in categories) or len(set(categories)) != len(categories) or not set(categories).issubset(features):
        raise ValueError("categorical_features deve ser uma lista de features finais distintas")
    
    for column, dtype in output_types(config, names).items():
        if column in features and ((dtype == "category") != (column in categories)):
            raise ValueError(f"Tipo de {column} incompatível com categorical_features")
        
    return names

def verify_run(directory):
    report = json.loads((directory / "run.json").read_text(encoding="utf-8"))

    if report.get("status") != "completed":
        raise ValueError("A execução de origem não foi concluída")
    
    if report.get("format_version") != 2:
        raise ValueError("Execução antiga de embeddings: gere as features novamente com outputs explícitos")
    
    for relative, expected in report["artifacts_sha256"].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or file_hash(path) != expected:
            raise ValueError(f"Artefato alterado/inválido: {relative}")
        
    return report

def run_models(config_path=None, *, stage="features", df=None, split_result=None,
               dry_run=False, from_run=None, sources_path=None, root=PATHS.root):
    if stage == "embeddings":
        stage = "features"

    if stage not in {"features", "train"}:
        raise ValueError("stage deve ser features ou train")
    
    config_path = config_path or root / "src/config/models.yaml"
    config = load_yaml(config_path)

    if config.get("schema_version") == 3:
        from src.models.features_pipeline import run_features

        if stage == "train":
            from src.models.fusion_pipeline import train_fusion, validate_fusion_config

            validate_fusion_config(config)
            if from_run is not None:
                return train_fusion(config_path, from_run=from_run, dry_run=dry_run, root=root)
            
            features_run = run_features(config_path, df=df, split_result=split_result,
                dry_run=dry_run, sources_path=sources_path, root=root)
            
            if dry_run:
                print("[Fusão / dry-run] Treinamento planejado após a geração; nenhuma rede executada")

                return None
            
            return train_fusion(config_path, from_run=features_run, root=root)
        
        if from_run is not None:
            raise ValueError("--from-run exige --stage train")
            
        return run_features(config_path, df=df, split_result=split_result,
            dry_run=dry_run, sources_path=sources_path, root=root)
    
    names = validate_architecture(config)

    global_cfg = config["global_config"]
    id_col, target = global_cfg["split"]["id_column"], global_cfg["target_col"]

    if target == "split" or id_col == "split" or target == id_col:
        raise ValueError("ID, alvo e split devem ter nomes distintos")
    
    seed = global_cfg["split"].get("seed", 42)
    generated = [spec["name"] for name in names for spec in config["modelos"][name]["outputs"]]
    features = config["multimodal"]["features"]
    prepared = {}

    parent = None
    if from_run is not None:
        if stage != "train":
            raise ValueError("--from-run só pode ser usado com --stage train")
        
        parent = project_path(from_run, root)
        verify_run(parent)
        old = load_yaml(parent / "models.yaml")

        if config["modelos"] != old["modelos"] or global_cfg != old["global_config"]:
            raise ValueError("Para reutilizar saídas, preserve modelos/global_config; altere apenas multimodal/reports")
        
        if names != old["multimodal"]["encoders"]:
            raise ValueError("Para reutilizar saídas, preserve multimodal.encoders e sua ordem")
        
        available = pd.read_parquet(parent / "multimodal.parquet")
        require_columns(available, features, "Multimodal")
    else:
        if df is None:
            df = pd.read_parquet(project_path(global_cfg["dataset"], root))

        if "split" in df or set(generated).intersection(df.columns):
            raise ValueError("Colunas geradas/split já existem na origem; não é permitido sobrescrever dados")
        
        require_columns(df, [c for c in features if c not in generated], "Features originais do multimodal")
        split_result = split_result or get_or_create_split(df, config, root=root, dry_run=dry_run)
        prepared = prepare_models(config_path, df=df, split_result=split_result, dry_run=dry_run, root=root)

    if dry_run:
        print(f"[Multimodal / dry-run] Colunas novas ({len(generated)}): {generated}; features finais: {features}; etapa: {stage}")

        return None

    report = RunReport(project_path(config.get("reports", {}).get("directory", "reports/runs"), root), config, root)
    directory = report.directory
    report.metadata.update(format_version=2, stage=stage, encoders=names, seed=seed)

    print(f"[Multimodal] Execução: {directory}")
    try:
        source_cfg = sources_path or root / "src/config/sources.yaml"

        if source_cfg.exists():
            shutil.copyfile(source_cfg, directory / "sources.yaml")

        if parent is not None:
            for filename in ["multimodal.parquet", "split.json", "encoder_metrics.json", "feature_schema.json", "sources.yaml"]:
                source = parent / filename

                if source.exists():
                    shutil.copyfile(source, directory / filename)

            shutil.copytree(parent / "encoders", directory / "encoders")
            fused = pd.read_parquet(directory / "multimodal.parquet")
            report.metadata["features_from_run"] = str(parent)
            
        else:
            if split_result is None:
                raise RuntimeError("Resultado da divisão não foi preparado")

            write_json(directory / "split.json", split_result.report)
            frames = split_result.frames
            metadata = pd.concat([frame.assign(split=part)
                                  for part, frame in frames.items()], ignore_index=True)
            
            blocks, encoder_metrics = {}, {}
            for name in names:
                cfg = config["modelos"][name]

                encoder = create_encoder(cfg, seed, root)
                item = prepared[name]
                
                if item.y is not None or any(p.y is not None for p in item.partitions.values()):
                    raise ValueError("Extrator recebeu alvos; role: encoder obrigatório")
                
                start = perf_counter()
                encoder.fit(item.partitions["train"].X, item.partitions["validation"].X)

                stats = {"fit_seconds": perf_counter() - start, "modality": cfg["modality"],
                         "target_used": False, "partitions": {}}
                
                part_tables = []
                for part in PARTS:
                    selected = item.partitions[part]
                    start = perf_counter()
                    vectors = encoder.transform(selected.X)
                    outputs = project_outputs(vectors, cfg, len(selected.X))
                    stats["partitions"][part] = {"rows": len(outputs), "generated_columns": outputs.columns.tolist(),
                                                   "internal_dimensions": vectors.shape[1],
                                                   "transform_seconds": perf_counter() - start}
                    
                    part_tables.append(feature_table(selected.ids, outputs, id_col))

                if len({tuple(f.columns) for f in part_tables}) != 1:
                    raise ValueError("Extrator mudou a dimensão entre conjuntos")
                
                blocks[name] = pd.concat(part_tables, ignore_index=True)
                save_encoder(encoder, directory / "encoders" / name)
                encoder_metrics[name] = stats

                print(f"[Extrator] {name}: {len(blocks[name])} linhas; colunas: {blocks[name].columns.tolist()[1:]}")

            fused = fuse_tables(metadata, blocks, id_col)
            fused.to_parquet(directory / "multimodal.parquet", index=False)

            write_json(directory / "encoder_metrics.json", encoder_metrics)

        features = config["multimodal"]["features"]
        require_columns(fused, features, "Multimodal")
        write_json(directory / "feature_schema.json", {"id_column": id_col, "target_column": target,
                                                       "features": features, "generated_columns": generated,
                                                       "generated_types": output_types(config, names), "encoders": names})
        
        report.metadata.update(rows=len(fused), generated_columns=len(generated), final_features=len(features),
                               dataset_sha256=file_hash(directory / "multimodal.parquet"))
        
        if stage == "train":
            parts = {part: fused.loc[fused["split"] == part] for part in PARTS}
            classifier = FusionClassifier(config["multimodal"], seed)
            
            start = perf_counter()
            classifier.fit(parts["train"][features], parts["train"][target],
                           parts["validation"][features], parts["validation"][target])
            
            report.metadata["classifier_fit_seconds"] = perf_counter() - start
            classifier.save(directory / "multimodal_model")
            report.metadata["classifier_input_dimensions"] = classifier.input_dimensions

            metrics = {}
            for part, frame in parts.items():
                start = perf_counter()
                probabilities = classifier.predict_proba(frame[features])
                metrics[part] = classification_results(directory, part, frame[id_col], frame[target],
                                  probabilities, classifier.labels.classes_.tolist(), perf_counter() - start)
                
            write_json(directory / "metrics.json", metrics)
            print(f"[Multimodal] F1 macro no teste: {metrics['test']['f1_macro']:.4f}")

        report.finish()
        
        return directory
    
    except Exception as exc:
        report.finish(error=exc)
        raise

def main():
    parser = argparse.ArgumentParser(description="Extratores e classificador multimodal")
    parser.add_argument("--config", default="src/config/models.yaml")
    parser.add_argument("--stage", choices=["features", "embeddings", "train"], default="features")
    parser.add_argument("--from-run", help="Reutiliza colunas e extratores de uma execução concluída")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        run_models(project_path(args.config), stage=args.stage, dry_run=args.dry_run, from_run=args.from_run)
    except (ValueError, KeyError, OSError, ImportError) as exc:
        parser.exit(1, f"[Erro] {exc}\n")

if __name__ == "__main__":
    main()
