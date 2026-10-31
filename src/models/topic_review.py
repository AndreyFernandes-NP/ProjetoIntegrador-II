"""
topic_review.py
---------------
Funções para comparar os tópicos com a nova classificação em uma amostra de validade.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
import yaml

from src.config import PATHS
from src.data.processor import load_yaml, project_path, require_columns
from src.data.validator import write_json
from src.models.features_pipeline import validate_features_config
from src.models.language import ContextualNLP

def review_topics(input_path: Path, config_path: Path, *, root: Path = PATHS.root,
                  split: str = "validation", limit: int = 200, dry_run: bool = False):
    """Reclassifica textos distintos e salva CSV editável, configuração e resumo."""
    config = load_yaml(project_path(config_path, root))
    names = validate_features_config(config)
    name = next((n for n in names if config["modelos"][n]["adapter"] == "contextual_nli"), None)
    if name is None:
        raise ValueError("A configuração atual usa similaridades semânticas e não gera main_topic. "
                         "Para revisar tópicos antigos, informe o models.yaml da execução legada.")

    cfg = config["modelos"][name]
    if cfg["topics"].get("scoring") != "entailment" or "fonte" not in cfg["topics"]:
        raise ValueError("A revisão isolada exige topics.scoring: entailment e topics.fonte")
    
    if split not in {"train", "validation", "test", "all"} or type(limit) is not int or limit < 0:
        raise ValueError("split inválido ou limit negativo")
    
    id_col = config["global_config"]["split"]["id_column"]
    data = pd.read_parquet(project_path(input_path, root))
    require_columns(data, [id_col, *cfg["features"]], "Revisão de tópicos")

    if split != "all":
        require_columns(data, ["split"], "Revisão por partição")
        data = data.loc[data["split"] == split]

    data = data.drop_duplicates(cfg["features"]).copy()
    seed = config["global_config"]["split"]["seed"]

    if limit and len(data) > limit:
        data = data.sample(n=limit, random_state=seed)

    if data.empty:
        raise ValueError("Nenhum chamado na seleção solicitada")
    
    if data[id_col].isna().any() or data[id_col].duplicated().any():
        raise ValueError("IDs devem ser preenchidos e únicos")
    
    if dry_run:
        print(f"[Tópicos / dry-run] {len(data)} textos distintos em {split}; sem downloads ou gravação")
        return None
    
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    directory = Path(root) / "reports" / "topics" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "models.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    summary = {"status": "running", "rows": len(data), "split": split, "seed": seed,
               "selection": "distinct feature rows", "accuracy": None}
    
    write_json(directory / "summary.json", summary)
    try:
        model = ContextualNLP(cfg, Path(root), seed).prepare_topics()
        selected = data.set_index(id_col)
        result = selected[cfg["features"]].copy()

        for col in ("main_topic", "semantic_margin"):
            if col in selected:
                result[f"old_{col}"] = selected[col]

        result = result.join(model.classify_topics(selected[cfg["features"]]))
        result["reviewed_topic"] = ""
        result.to_csv(directory / "topic_review.csv", index=True, encoding="utf-8-sig")

        summary.update(status="completed", model=model.details,
                       new_distribution=result["main_topic"].value_counts().to_dict(),
                       decisions=result["topic_decision"].value_counts().to_dict(),
                       coverage=float(result["topic_decision"].eq("accepted").mean()),
                       rejection=cfg["topics"]["rejection"])
        
        if "old_main_topic" in result:
            summary["old_distribution"] = result["old_main_topic"].value_counts().to_dict()

        write_json(directory / "summary.json", summary)
    except Exception as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(directory / "summary.json", summary)

        raise

    print(f"[Tópicos] Revisão: {directory / 'topic_review.csv'}")
    print(f"[Tópicos] Cobertura: {summary['coverage']:.1%}; acurácia ainda não medida")

    return directory

def main() -> None:
    """Executa revisão de tópicos, com simulação explícita sem download."""
    parser = argparse.ArgumentParser(description="Revisão isolada de main_topic")
    parser.add_argument("--input", required=True, help="Parquet enriquecido a revisar")
    parser.add_argument("--config", default="src/config/models.yaml")
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="validation")
    parser.add_argument("--limit", type=int, default=200, help="Textos distintos; 0 usa todos da partição")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        review_topics(Path(args.input), Path(args.config), split=args.split, limit=args.limit, dry_run=args.dry_run)

    except (ValueError, KeyError, OSError, ImportError) as exc:
        parser.exit(1, f"[Erro] {exc}\n")

if __name__ == "__main__":
    main()
