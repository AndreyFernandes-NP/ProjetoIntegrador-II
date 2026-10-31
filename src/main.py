"""Projeto Integrador II - menu interativo para dados, modelos e relatórios.

Menu básico interativo que aborda o projeto inteiro, desde a pipeline inicial até o treinamento de modelos.
O menu valida pré-requisitos antes de invocar os módulos oficiais do projeto.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "src/config/sources.yaml"
MODELS = ROOT / "src/config/models.yaml"

class SetupError(Exception):
    """Configuração incompleta ou inválida antes de iniciar uma etapa."""

def local_path(value: str | Path) -> Path:
    """Interpreta caminhos de configuração em relação à raiz do repositório."""
    path = Path(value).expanduser()

    return (path if path.is_absolute() else ROOT / path).resolve()

def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    
    except ValueError:

        return str(path)

def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SetupError(f"Arquivo ausente: {relative(path)}")
    
    try:
        import yaml

    except ImportError as exc:
        raise SetupError("Biblioteca PyYAML ausente. Instale as dependências de requirements.txt.") from exc
    
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))

    except (OSError, yaml.YAMLError) as exc:
        raise SetupError(f"Erro ao ler {relative(path)}: {exc}") from exc
    
    if not isinstance(data, dict):
        raise SetupError(f"O YAML está vazio ou inválido: {relative(path)}")
    
    return data

def get_config_path(config: dict[str, Any], *keys: str) -> Path:
    value: Any = config

    try:
        for key in keys:
            value = value[key]

    except (KeyError, TypeError) as exc:
        raise SetupError(f"Falta o campo {'.'.join(keys)} no YAML.") from exc
    
    if not isinstance(value, str) or not value.strip():
        raise SetupError(f"Campo inválido: {'.'.join(keys)} (esperado caminho).")
    
    return local_path(value)

def pause() -> None:
    try:
        input("\nPressione Enter para voltar ao menu principal...")

    except EOFError:
        pass

def choose(prompt: str) -> str:
    try:
        return input(prompt).strip()
    
    except (EOFError, KeyboardInterrupt):
        print()
        return "0"

def header(title: str) -> None:
    print("\n" + "=" * 67)
    print("  " + title)
    print("=" * 67)

def missing_file(path: Path, explanation: str, issues: list[str]) -> None:
    if not path.is_file():
        issues.append(f"{relative(path)} — {explanation}")

def packages_for(stage: str) -> dict[str, str]:
    common = {"yaml": "PyYAML", "pandas": "pandas", "numpy": "numpy"}
    table = {
        "process_only": {**common, "pyarrow": "pyarrow"},
        "data": {**common, "pyarrow": "pyarrow", "sklearn": "scikit-learn", "PIL": "Pillow"},
        "download": {**common, "torch": "torch", "transformers": "transformers", "huggingface_hub": "huggingface-hub", "PIL": "Pillow"},
        "features": {**common, "pyarrow": "pyarrow", "torch": "torch", "sklearn": "scikit-learn", "transformers": "transformers", "huggingface_hub": "huggingface-hub", "PIL": "Pillow"},
        "fusion": {**common, "pyarrow": "pyarrow", "torch": "torch", "sklearn": "scikit-learn", "threadpoolctl": "threadpoolctl"},
        "predict": {**common, "pyarrow": "pyarrow", "torch": "torch", "sklearn": "scikit-learn", "transformers": "transformers", "PIL": "Pillow"},
    }

    return table.get(stage, {})

def check_packages(stage: str, issues: list[str]) -> None:
    not_installed = [name for module, name in packages_for(stage).items()
                     if importlib.util.find_spec(module) is None]
    
    if not_installed:
        issues.append("Bibliotecas Python ausentes: " + ", ".join(not_installed)
                      + ". Consulte requirements.txt.")

def check_image_paths_csv(csv_file: Path, base: Path, column: str, issues: list[str], *, encoding: str = "utf-8-sig", delimiter: str = ",") -> None:
    """Checa imagens referenciadas por CSV sem carregar grandes redes nem Pandas."""
    if not csv_file.is_file():
        return
    
    invalid: list[str] = []
    missing_count = 0

    try:
        with csv_file.open(newline="", encoding=encoding) as stream:
            reader = csv.DictReader(stream, delimiter=delimiter)
            if column not in (reader.fieldnames or []):
                issues.append(f"{relative(csv_file)} não possui a coluna '{column}'.")

                return
            
            seen: set[str] = set()
            for record in reader:
                value = (record.get(column) or "").strip()

                if value in seen:
                    continue

                seen.add(value)
                if not value or not local_image_path(value, base).is_file():
                    missing_count += 1

                    if len(invalid) < 4:
                        invalid.append(value or "<vazio>")

    except (OSError, UnicodeError, LookupError, csv.Error) as exc:
        issues.append(f"Não foi possível verificar imagens em {relative(csv_file)}: {exc}")

        return
    
    if missing_count:
        issues.append(f"{relative(csv_file)}: {missing_count} imagem(ns) ausente(s) "
                      f"(base {relative(base)}), exemplos: {', '.join(invalid)}")

def local_image_path(value: str, base: Path) -> Path:
    path = Path(value)

    return (path if path.is_absolute() else base / path).resolve()

def check_processed_images(dataset: Path, base: Path, column: str, issues: list[str]) -> None:
    if not dataset.is_file():
        return
    
    try:
        import pandas as pd

        data = pd.read_parquet(dataset, columns=[column])
        values = data[column].drop_duplicates().tolist()

        bad = [str(v) for v in values if pd.isna(v) or not local_image_path(str(v), base).is_file()]
        if bad:
            issues.append(f"{relative(dataset)} referencia {len(bad)} imagem(ns) inexistente(s); "
                          f"ex.: {', '.join(bad[:4])}")
    except ImportError:
        pass

    except Exception as exc:  # Dataset ilegível
        issues.append(f"Não foi possível verificar a coluna de imagens em {relative(dataset)}: {exc}")

def get_image_settings(config: dict[str, Any]) -> tuple[Path, str, dict[str, Any]] | None:
    for stage in (config.get("enrichment", {}) or {}).get("stages", []):
        item = config.get("modelos", {}).get(stage, {})

        if item.get("adapter") == "blip_captioning" and item.get("enabled", False):
            base = get_config_path(item, "preprocessing", "base_dir")
            column = item.get("features", ["image_path"])[0]

            return base, column, item
        
    return None

def check_models_related(config: dict[str, Any], issues: list[str], *, annotated: bool, processed_images: bool, source_images: bool, sources_cfg: dict[str, Any] | None = None) -> None:
    image_cfg = get_image_settings(config)
    if image_cfg is None:
        return
    
    base, column, item = image_cfg
    if annotated and item.get("training", {}).get("enabled"):
        file = get_config_path(item, "training", "captions_file")

        missing_file(file, "legendas para ajuste do BLIP; gere/adicione image_captions.csv", issues)
        if file.is_file():
            check_image_paths_csv(file, base, "image_path", issues)

    if processed_images:
        dataset = get_config_path(config, "global_config", "dataset")
        check_processed_images(dataset, base, column, issues)

    if source_images and sources_cfg:
        source = get_config_path(sources_cfg, "dataset", "entrada")

        if source.suffix.lower() == ".csv":
            reading = (sources_cfg.get("dataset", {}).get("leitura") or {})

            check_image_paths_csv(source, base, column, issues,
                                  encoding=reading.get("encoding", "utf-8-sig"), delimiter=reading.get("sep", ","))
        # Se for Parquet, o validador/splitter do pipeline fará a checagem.

def check_requirements(stage: str, *, feature_run: Path | None = None,
                       fusion_run: Path | None = None, input_file: Path | None = None,
                       raw: bool = False) -> list[str]:
    """Não executa o pipeline; identifica arquivos reais necessários a esta ação."""
    issues: list[str] = []

    modules = {
        "process_only": ["src/data/pipeline.py"],
        "data": ["src/data/pipeline.py", "src/data/splitter.py"],
        "download": ["src/models/features_pipeline.py"],
        "features": ["src/models/features_pipeline.py", "src/models/vision.py", "src/models/language.py", "src/models/profiles.py"],
        "fusion": ["src/models/fusion_pipeline.py", "src/models/fusion.py"],
        "all": ["src/data/pipeline.py", "src/models/pipeline.py", "src/models/features_pipeline.py", "src/models/fusion_pipeline.py"],
        "predict": ["src/models/fusion_predict.py", "src/models/fusion_pipeline.py"],
    }

    for path in modules.get(stage, []):
        missing_file(ROOT / path, "módulo do projeto necessário nesta etapa", issues)

    check_packages("features" if stage == "all" else stage, issues)
    if stage == "all":
        check_packages("fusion", issues)

    sources_cfg: dict[str, Any] | None = None
    models_cfg: dict[str, Any] | None = None

    try:
        if stage in {"process_only", "data", "all"}:
            sources_cfg = load_yaml(SOURCES)
            source = get_config_path(sources_cfg, "dataset", "entrada")

            missing_file(source, "dataset original configurado em sources.yaml", issues)
        if stage in {"data", "download", "features", "fusion", "all"}:
            models_cfg = load_yaml(MODELS)

        if models_cfg and stage in {"features", "fusion"}:
            dataset = get_config_path(models_cfg, "global_config", "dataset")

            if stage == "features":
                missing_file(dataset, "processe os dados primeiro (menu Dados)", issues)

        if models_cfg and stage in {"features", "all"}:
            check_models_related(models_cfg, issues, annotated=True,
                                 processed_images=(stage == "features"), source_images=(stage == "all"),
                                 sources_cfg=sources_cfg)
            
        if feature_run:
            for name in ("run.json", "enriched.parquet", "models.yaml", "feature_schema.json", "split.json"):
                missing_file(feature_run / name, "artefato de features requerido para a fusão", issues)

        if fusion_run:
            for name in ("run.json", "models.yaml"):
                missing_file(fusion_run / name, "artefato de treinamento necessário à predição", issues)

            try:
                report = json.loads((fusion_run / "run.json").read_text(encoding="utf-8"))

                if report.get("status") != "completed" or report.get("format_version") != 4:
                    issues.append("Execução multimodal não está concluída (formato 4).")

                model_dir = local_path(report["model_directory"])
                if not model_dir.is_dir():
                    issues.append(f"Pesos/artefatos do classificador ausentes: {relative(model_dir)}")

                if raw:
                    origin = report.get("features_run")

                    if not origin:
                        issues.append("Essa fusão não possui origem de features reutilizável; use entrada enriquecida.")

                    else:
                        path = local_path(origin)

                        for name in ("run.json", "enriched.parquet", "models.yaml"):
                            missing_file(path / name, "geradores originais necessários a --raw", issues)

                        original_models = ROOT / "models/runs" / path.name
                        if not original_models.is_dir():
                            issues.append(f"Pesos dos geradores ausentes: {relative(original_models)}")

            except (OSError, ValueError, KeyError) as exc:
                issues.append(f"Manifesto da fusão ilegível: {exc}")

        if input_file:
            missing_file(input_file, "arquivo de entrada da predição", issues)

            if input_file.suffix.lower() not in {".csv", ".parquet"}:
                issues.append("Entrada da predição deve ser CSV ou Parquet.")

    except (SetupError, KeyError, TypeError, IndexError, ValueError) as exc:
        issues.append(f"Configuração incompleta ou inválida: {exc}")

    return list(dict.fromkeys(issues))

def blocked(issues: list[str]) -> bool:
    if not issues:
        return False
    
    print("\n[BLOQUEADO] A etapa não foi iniciada. Verifique:")
    for issue in issues:
        print(f"  - {issue}")
    print("\nNenhum comando foi executado. Resolva as pendências e tente novamente.")
    pause()

    return True

def confirm(message: str, keyword: str = "S") -> bool:
    answer = choose(f"{message} Digite {keyword} para confirmar: ")

    return answer.upper() == keyword

def execute(stage: str, module: str, args: list[str], *, feature_run: Path | None = None,
            fusion_run: Path | None = None, input_file: Path | None = None, raw: bool = False) -> None:
    issues = check_requirements(stage, feature_run=feature_run, fusion_run=fusion_run,
                                input_file=input_file, raw=raw)
    
    if blocked(issues):
        return
    
    print("\nComando: " + " ".join([sys.executable, "-m", module, *args]))
    if not confirm("Executar essa etapa?"):
        print("Etapa cancelada.")

        pause()
        return
    
    print("\n" + "-" * 67, flush=True)

    try:
        result = subprocess.run([sys.executable, "-m", module, *args], cwd=ROOT, check=False)

        print("\n" + "-" * 67)
        print("[CONCLUÍDO]" if result.returncode == 0 else f"[FALHOU] Código de saída: {result.returncode}")

        if result.returncode != 0:
            print("A mensagem do pipeline acima contém os detalhes do erro.")

    except KeyboardInterrupt:
        print("\n[INTERROMPIDO] A execução foi interrompida pelo usuário.")

    except OSError as exc:
        print(f"\n[ERRO] Não foi possível iniciar o processo: {exc}")

    pause()

def config_report_dir(section: str, fallback: str) -> Path:
    try:
        config = load_yaml(MODELS)

        if section == "features":
            value = (config.get("reports") or {}).get("directory", fallback)

        else:
            value = (config.get("multimodal") or {}).get("reports_directory", fallback)

        return local_path(value)
    
    except SetupError:
        return ROOT / fallback

def completed_runs(kind: str) -> list[tuple[Path, dict[str, Any]]]:
    base = config_report_dir(kind, "reports/runs" if kind == "features" else "reports/fusion")
    found: list[tuple[Path, dict[str, Any]]] = []

    if not base.is_dir():
        return found
    
    for path in base.iterdir():
        manifest = path / "run.json"

        if not manifest.is_file():
            continue

        try:
            report = json.loads(manifest.read_text(encoding="utf-8"))

        except (OSError, ValueError):
            continue

        version = 3 if kind == "features" else 4
        if report.get("status") != "completed" or report.get("format_version") != version:
            continue

        required = ("enriched.parquet", "models.yaml", "feature_schema.json", "split.json") if kind == "features" else ("metrics.json", "models.yaml")

        if all((path / name).is_file() for name in required):
            found.append((path, report))

    return sorted(found, key=lambda pair: pair[0].name, reverse=True)

def select_run(kind: str) -> Path | None:
    runs = completed_runs(kind)

    header("Execuções concluídas: " + ("features" if kind == "features" else "multimodal"))
    if not runs:
        print("Nenhuma execução concluída com os arquivos essenciais foi encontrada.")
        print("Use primeiro a etapa correspondente no menu Modelos.")
        pause()

        return None
    
    for i, (path, report) in enumerate(runs[:15], 1):
        extra = ""
        if kind == "fusion":
            metric_path = path / "metrics.json"
            try:
                metrics = json.loads(metric_path.read_text(encoding="utf-8"))
                test = metrics.get("test", {})
                extra = f"  teste: acc={test.get('accuracy', 0):.3f}; F1={test.get('f1_macro', 0):.3f}"

            except (OSError, ValueError, TypeError):
                pass

        print(f" {i:2}. {path.name}{extra}")

    print("  0. Cancelar e voltar ao menu principal")
    selection = choose("Escolha a execução: ")

    if selection.isdigit() and 1 <= int(selection) <= min(15, len(runs)):
        return runs[int(selection) - 1][0]
    
    return None

def data_menu() -> None:
    header("DADOS E DIVISÃO")
    print("  1. Verificar processamento e split (sem salvar)")
    print("  2. Processar CSV para Parquet (sem recriar split)")
    print("  3. Processar, validar e criar/reutilizar split")
    print("  4. Recriar split deliberadamente (altera a divisão)")
    print("  0. Voltar ao menu principal")
    print("")

    option = choose("Escolha: ")
    if option == "1":
        execute("data", "src.data.pipeline", ["--dry-run"])

    elif option == "2":
        execute("process_only", "src.data.pipeline", ["--skip-split"])

    elif option == "3":
        execute("data", "src.data.pipeline", [])

    elif option == "4":
        if confirm("ATENÇÃO: o split antigo poderá deixar de ser reproduzível.", "RECRIAR"):
            execute("data", "src.data.pipeline", ["--rebuild-split"])

    elif option != "0":

        print("Opção inválida.")
        pause()

def models_menu() -> None:
    header("MODELOS E PIPELINE")
    print("  1. Baixar/preparar modelos pré-treinados")
    print("  2. Gerar features a partir de data/processed (visão, PLN, tabular)")
    print("  3. Treinar multimodal a partir de uma execução de features existente")
    print("  4. Pipeline completa: processar + features + multimodal")
    print("  0. Voltar ao menu principal")
    print("")

    option = choose("Escolha: ")
    if option == "1":
        execute("download", "src.models.features_pipeline", ["--download-only"])

    elif option == "2":
        execute("features", "src.models.features_pipeline", [])

    elif option == "3":
        run = select_run("features")

        if run is not None:
            execute("fusion", "src.models.fusion_pipeline", ["--from-run", str(run)], feature_run=run)

    elif option == "4":
        execute("all", "src.data.pipeline", ["--train-multimodal"])

    elif option != "0":
        print("Opção inválida.")
        pause()

def prediction_menu() -> None:
    run = select_run("fusion")

    if run is None:
        return
    
    header("PREDIÇÃO COM MODELO SALVO")
    print(f"Modelo selecionado: {run.name}")
    print("  1. Entrada já enriquecida (com features geradas)")
    print("  2. Entrada apenas processada (reaplica geradores salvos)")
    print("  0. Voltar ao menu principal")
    print("")

    mode = choose("Escolha: ")
    if mode not in {"1", "2"}:
        return
    
    raw = mode == "2"
    default = "data/processed/chamados.parquet" if raw else ""
    value = choose(f"Arquivo de entrada (.csv/.parquet){f' [{default}]' if default else ''}: ") or default

    if not value:
        print("Nenhum arquivo escolhido.")
        pause()

        return
    
    input_file = local_path(value)
    output_default = "reports/predictions/predicoes_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
    destination = local_path(choose(f"Arquivo de saída [{output_default}]: ") or output_default)

    if destination.suffix.lower() not in {".csv", ".parquet"} or destination == input_file:
        print("[BLOQUEADO] A saída deve ser .csv/.parquet e diferente da entrada.")
        pause()

        return
    
    if destination.exists() and not confirm(f"A saída {relative(destination)} já existe e será substituída.", "SUBSTITUIR"):
        pause()

        return
    
    args = ["--run", str(run), "--input", str(input_file), "--output", str(destination)]
    if raw:
        args.append("--raw")

    execute("predict", "src.models.fusion_predict", args, fusion_run=run, input_file=input_file, raw=raw)

def reports_menu() -> None:
    run = select_run("fusion")
    if run is None:

        return
    
    header("RELATÓRIO DO MULTIMODAL")
    try:
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))

        print(f"Execução: {relative(run)}\n")
        print(f"{'Partição':<15}{'Accuracy':>14}{'F1 macro':>14}{'Balanced acc.':>17}")
        print("-" * 60)

        for name in ("train", "validation", "test"):
            part = metrics.get(name, {})
            print(f"{name:<15}{part.get('accuracy', 0):>14.4f}{part.get('f1_macro', 0):>14.4f}{part.get('balanced_accuracy', 0):>17.4f}")

        print("\nArquivos adicionais:")
        for name in ("best_params.yaml", "trials.csv", "validation_confusion.csv", "test_confusion.csv"):
            if (run / name).is_file():
                print(f"  {relative(run / name)}")

    except (OSError, ValueError, TypeError) as exc:
        print(f"Não foi possível ler este relatório: {exc}")

    pause()

def diagnostic_menu() -> None:
    header("DIAGNÓSTICO DE PRÉ-REQUISITOS")

    for name, stage in (("Processamento simples", "process_only"),
                        ("Processamento com split", "data"),
                        ("Download dos modelos", "download"),
                        ("Geração de features", "features"),
                        ("Pipeline completa", "all")):
        print(f"\n{name}:")
        issues = check_requirements(stage)

        if not issues:
            print("  [OK] Arquivos e bibliotecas essenciais encontrados.")

        else:
            for item in issues[:8]:
                print(f"  [FALTA] {item}")

            if len(issues) > 8:
                print(f"  ... mais {len(issues) - 8} pendência(s)")

    print(f"\nExecuções de features concluídas: {len(completed_runs('features'))}")
    print(f"Treinamentos multimodais concluídos: {len(completed_runs('fusion'))}")
    print("\nObservação: verificações de conteúdo, hashes, schema e consistência são feitas pelos pipelines.")

    pause()

def main() -> None:
    try:
        while True:
            header("PROJETO INTEGRADOR II  |  MENU PRINCIPAL")
            print("  1. Dados: validar, processar e dividir")
            print("  2. Modelos: downloads, features e treinamento multimodal")
            print("  3. Predição com modelo treinado")
            print("  4. Consultar métricas e relatórios")
            print("  5. Diagnosticar arquivos e dependências")
            print("  0. Sair\n")

            option = choose("Escolha uma área: ")

            if option == "1":
                data_menu()

            elif option == "2":
                models_menu()

            elif option == "3":
                prediction_menu()

            elif option == "4":
                reports_menu()

            elif option == "5":
                diagnostic_menu()

            elif option == "0":
                print("Programa encerrado.")
                break
            else:
                print("Opção inválida. Escolha uma das áreas exibidas.")

                pause()

    except KeyboardInterrupt:
        print("\nPrograma encerrado pelo usuário.")

if __name__ == "__main__":
    main()
