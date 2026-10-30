"""
fusion_predict.py
-----------------
Classifica chamados com o multimodal salvo, a partir de features enriquecidas ou
reutilizando os geradores da execução de origem.
"""
import argparse
from pathlib import Path

import pandas as pd

from src.config import PATHS
from src.data.processor import project_path, load_yaml, require_columns
from src.models.fusion import EarlyFusionClassifier
from src.models.fusion_pipeline import verify_fusion_run

def predict_targets(data: pd.DataFrame, run_directory: Path, *, raw: bool = False, root: Path = PATHS.root) -> pd.DataFrame:
    """Retorna ID, target previsto, score e probabilidades.

    raw=True recebe dados já tratados pelo processor, ainda sem outputs dos
    geradores. Exige uma fusão treinada com --from-run e os seus artefatos originais.
    """
    directory = project_path(run_directory, root)

    report, artifacts = verify_fusion_run(directory, root)
    config = load_yaml(directory / "models.yaml")
    id_col = config["global_config"]["split"]["id_column"]

    require_columns(data, [id_col], "Predição")
    if data.empty or data[id_col].isna().any() or data[id_col].astype(str).str.strip().eq("").any() or data[id_col].astype(str).duplicated().any():
        raise ValueError("Predição exige chamados com IDs preenchidos e únicos")
    
    work = data.copy().reset_index(drop=True)
    if raw:
        if not report.get("features_run"):
            raise ValueError("Fusão sem origem verificável, forneça features já enriquecidas")
        
        from src.models.features_pipeline import enrich_new
        work = enrich_new(work, project_path(report["features_run"], root), root=root)

    model = EarlyFusionClassifier.restore(artifacts)
    probabilities = model.predict_proba(work)
    result = work[[id_col]].copy()
    result["predicted_target"] = model.classes[probabilities.argmax(axis=1)]
    result["prediction_score"] = probabilities.max(axis=1)

    for i, label in enumerate(model.classes):
        result[f"probability__{label}"] = probabilities[:, i]

    return result

def main() -> None:
    """Executa inferência e salva CSV ou Parquet em arquivo distinto da entrada."""
    parser = argparse.ArgumentParser(description="Predição do target com fusão salva")
    parser.add_argument("--run", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw", action="store_true", help="Gera features usando os artefatos originais; não refaz treino")
    args = parser.parse_args()
    try:
        source, output = project_path(args.input), project_path(args.output)
        if source == output or output.suffix.lower() not in {".csv", ".parquet"}:
            raise ValueError("Saída deve ser CSV/Parquet diferente da entrada")
        
        if source.suffix.lower() == ".parquet":
            data = pd.read_parquet(source)

        elif source.suffix.lower() == ".csv":
            cfg = load_yaml(project_path(args.run) / "models.yaml")
            data = pd.read_csv(source, dtype={cfg["global_config"]["split"]["id_column"]: "string"})

        else:
            raise ValueError("Entrada deve ser CSV ou Parquet")
        
        result = predict_targets(data, project_path(args.run), raw=args.raw)
        output.parent.mkdir(parents=True, exist_ok=True)

        if output.suffix.lower() == ".parquet":
            result.to_parquet(output, index=False)

        else:
            result.to_csv(output, index=False, encoding="utf-8-sig")

        print(f"[Predição] {len(result)} chamados: {output}")

    except (ValueError, KeyError, OSError, ImportError) as exc:
        parser.exit(1, f"[Erro] {exc}\n")

if __name__ == "__main__":
    main()
