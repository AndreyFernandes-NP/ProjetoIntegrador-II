"""
pipeline.py
-----------
Pipeline principal do projeto, que processa e valida o dataset, cria/reutiliza a divisão
de dados e opcionalmente prepara os modelos habilitados do arquivo de config.

- Formas de execução:
    - python -m src.data.pipeline               // processa, valida e salva o dataset em .parquet e cria/reutiliza a divisão
    - python -m src.data.pipeline --dry-run     // simula processamento, qualidade e divisão sem salvar ou baixar modelos
    - python -m src.data.pipeline --models      // também seleciona features dos modelos habilitados e suas partições
    - python -m src.data.pipeline --dry-run --models
    - python -m src.data.pipeline --load-models
    - python -m src.data.pipeline --model <nome_do_modelo>
    - python -m src.data.pipeline --skip-split  // processa e valida sem dividir os dados
    - python -m src.data.pipeline --rebuild-split // recria a divisão após alterações de dados/configuração
"""

import argparse

from src.data.processor import load_yaml, process_dataset, project_path
from src.models.loader import prepare_models
from src.data.splitter import get_or_create_split

def main() -> None:
    """Processa os dados e executa as etapas opcionais conforme a versão do YAML."""
    parser = argparse.ArgumentParser(description="Pipeline Projeto Integrador II")
    parser.add_argument("--sources", default="src/config/sources.yaml")
    parser.add_argument("--model-config", default="src/config/models.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Valida sem salvar ou baixar modelos")
    parser.add_argument("--models", action="store_true", help="Seleciona features dos modelos habilitados")
    parser.add_argument("--load-models", action="store_true", help="Também baixa/carrega os modelos")
    parser.add_argument("--model", help="Limita a seleção a um experimento habilitado")
    parser.add_argument("--skip-split", action="store_true", help="Executa sem criar/reutilizar divisões")
    parser.add_argument("--rebuild-split", action="store_true", help="Recria deliberadamente o manifesto de divisão")
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument("--features", "--embeddings", dest="features", action="store_true", help="Gera somente as colunas declaradas em outputs")
    stages.add_argument("--train-multimodal", action="store_true", help="Também treina e avalia o classificador final")
    args = parser.parse_args()

    if args.dry_run and args.load_models:
        parser.error("--dry-run não pode ser usado com --load-models")

    if args.skip_split and args.rebuild_split:
        parser.error("--skip-split não pode ser usado com --rebuild-split")
    if (args.features or args.train_multimodal) and (args.skip_split or args.model is not None):
        parser.error("A pipeline multimodal exige divisão e os três extratores; não use --skip-split ou --model")

    sources_path = project_path(args.sources)
    models_path = project_path(args.model_config)
    with_models = args.models or args.load_models or args.model is not None or args.features or args.train_multimodal

    try:
        model_config = load_yaml(models_path) if with_models or not args.skip_split else None
        current_features = model_config is not None and model_config.get("schema_version") == 3
        names = []

        if model_config is not None and current_features and with_models:
            from src.models.features_pipeline import validate_features_config, download_feature_models

            names = validate_features_config(model_config)

            if args.train_multimodal:
                from src.models.fusion_pipeline import validate_fusion_config

                validate_fusion_config(model_config)
                
            if args.model is not None:
                parser.error("Schema 3 usa as etapas em sequência; remova --model")

        if model_config is not None:
            destination = load_yaml(sources_path)["dataset"]["saida"]
            model_dataset = model_config["global_config"]["dataset"]

            if project_path(destination) != project_path(model_dataset):
                parser.error("A saida de sources.yaml deve coincidir com o dataset de models.yaml")

        df = process_dataset(sources_path, dry_run=args.dry_run)

        split = None
        if not args.skip_split:
            assert model_config is not None
            split = get_or_create_split(df, model_config, dry_run=args.dry_run, rebuild=args.rebuild_split)

        if with_models and current_features:
            assert model_config is not None

            for name in names:
                item = model_config["modelos"][name]

                print(f"[Pipeline] {name}: features={item['features']}; outputs={[o['name'] for o in item['outputs']]}")

            if args.load_models:
                from src.models.features_pipeline import download_feature_models
                
                download_feature_models(model_config)

        if with_models and not current_features:
            prepared = prepare_models(models_path, df=df, load=args.load_models, only=args.model,
                                      use_split=not args.skip_split, split_result=split, dry_run=args.dry_run)

            for name, item in prepared.items():
                state = "adaptador_criado" if item.config.get("role") == "encoder" else "carregado"
                print(f"[Pipeline] {name}: features={item.X.columns.tolist()}, target={item.y.name if item.y is not None else None}, {state}={item.model is not None}")

            if not prepared:
                print("[Pipeline] Nenhum modelo habilitado")

        if args.features or args.train_multimodal:
            from src.models.pipeline import run_models
            run_models(models_path, stage="train" if args.train_multimodal else "features",
                       df=df, split_result=split, dry_run=args.dry_run, sources_path=sources_path)

    except (ValueError, KeyError, FileNotFoundError, ImportError) as exc:
        parser.exit(1, f"[Erro] {exc}\n")

if __name__ == "__main__":
    main()
