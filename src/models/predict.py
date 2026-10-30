"""
predict.py
----------
Carrega os modelos e pré-processadores salvos para gerar as colunas intermediárias
e prever o target de novos registros, sem reajustar os modelos nem exigir o target na entrada.
"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.config import PATHS
from src.data.processor import load_yaml, project_path, require_columns
from src.models.encoders import restore_encoder
from src.models.multimodal import FusionClassifier, feature_table, fuse_tables
from src.models.pipeline import verify_run, validate_architecture
from src.models.outputs import project_outputs


def predict_new(df, run_directory, *, root=PATHS.root):
    run_directory = project_path(run_directory, root)
    verify_run(run_directory)

    if not (run_directory / "multimodal_model/model.keras").exists():
        raise ValueError("A execução contém apenas features; treine o multimodal primeiro")
    
    config = load_yaml(run_directory / "models.yaml")
    names = validate_architecture(config)
    id_col = config["global_config"]["split"]["id_column"]

    require_columns(df, [id_col], "Inferência")
    generated = [spec["name"] for name in names for spec in config["modelos"][name]["outputs"]]

    if set(generated).intersection(df.columns):
        raise ValueError("A entrada já contém outputs gerados; forneça somente os dados de origem")
    
    originals = [c for c in config["multimodal"]["features"] if c not in generated]
    require_columns(df, originals, "Features originais do multimodal")

    blocks = {}
    for name in names:
        encoder = restore_encoder(run_directory / "encoders" / name, root)
        features = encoder.config["features"]
        require_columns(df, features, name)
        vectors = encoder.transform(df[features])
        outputs = project_outputs(vectors, encoder.config, len(df))
        blocks[name] = feature_table(df[id_col], outputs, id_col)

    fused = fuse_tables(df[[id_col] + originals], blocks, id_col)
    classifier = FusionClassifier.restore(run_directory / "multimodal_model")
    probabilities = classifier.predict_proba(fused[classifier.features])
    result = fused[[id_col]].copy()
    result["predicted_target"] = classifier.labels.classes_[probabilities.argmax(axis=1)]

    for index, label in enumerate(classifier.labels.classes_):
        result[f"probability__{label}"] = probabilities[:, index]

    return result

def main():
    parser = argparse.ArgumentParser(description="Prever alvo de chamados novos, sem reajustar modelos")
    parser.add_argument("--run", required=True)
    parser.add_argument("--input", required=True, help="CSV ou Parquet com as features já tratadas; não exige target")
    parser.add_argument("--output", required=True, help="CSV de predições")
    args = parser.parse_args()
    source, destination = project_path(args.input), project_path(args.output)

    if source == destination:
        parser.error("Entrada e saída devem ser diferentes")

    if destination.suffix.lower() != ".csv":
        parser.error("A saída deve ser .csv")

    try:
        if source.suffix.lower() == ".parquet":
            df = pd.read_parquet(source)

        elif source.suffix.lower() == ".csv":
            config = load_yaml(project_path(args.run) / "models.yaml")
            id_col = config["global_config"]["split"]["id_column"]
            df = pd.read_csv(source, dtype={id_col: "string"})

        else:
            raise ValueError("Entrada deve ser CSV ou Parquet")
        
        start = perf_counter()
        result = predict_new(df, args.run)
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(destination, index=False, encoding="utf-8-sig")
        print(f"[Inferência] {len(result)} chamados em {perf_counter() - start:.2f}s: {destination}")

    except (ValueError, KeyError, OSError, ImportError) as exc:
        parser.exit(1, f"[Erro] {exc}\n")

if __name__ == "__main__":
    main()
