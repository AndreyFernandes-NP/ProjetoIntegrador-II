"""
processor.py
------------
Funções de processamento pro dataset, primeiro carrega a configuração do sources.yaml,
depois aplica as transformações, valida a qualidade e finalmente salva o resultado em parquet.
"""
from pathlib import Path

import pandas as pd
import yaml

from src.config import PATHS
from src.data.validator import validate_quality

def project_path(value: str | Path, root: Path = PATHS.root) -> Path:
    path = Path(value)

    return (path if path.is_absolute() else root / path).resolve()

def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError(f"Configuração vazia ou inválida: {path}")
    
    return config

def require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{context}: colunas ausentes: {missing}")

def cast_column(series: pd.Series, rule: dict) -> pd.Series:
    dtype = rule["cast"]
    errors = rule.get("errors", "raise")

    if errors not in {"raise", "coerce"}:
        raise ValueError("errors deve ser 'raise' ou 'coerce'")
    
    if dtype in {"string", "category"}:
        return series.astype(dtype)
    
    if dtype == "datetime":
        return pd.to_datetime(series, format=rule.get("format"), errors=errors)
    
    if dtype in {"Int64", "Float64"}:
        values = series

        if not pd.api.types.is_numeric_dtype(series):
            values = series.astype("string").str.strip()
            decimal = rule.get("decimal", ".")
            thousands = rule.get("thousands")

            if not isinstance(decimal, str) or len(decimal) != 1:
                raise ValueError("decimal deve ser um caractere")
            
            if thousands is not None:
                if not isinstance(thousands, str) or len(thousands) != 1 or thousands == decimal:
                    raise ValueError("thousands deve ser um caractere diferente de decimal")
                
                values = values.str.replace(thousands, "", regex=False)

            if decimal != ".":
                values = values.str.replace(decimal, ".", regex=False)

        result = pd.to_numeric(values, errors=errors)
        invalid = result.isin([float("inf"), float("-inf")])

        if dtype == "Int64":
            invalid |= result.notna() & (result % 1 != 0)

        if invalid.any() and errors == "raise":
            raise ValueError(f"Valores incompatíveis com {dtype}: {series[invalid].head().tolist()}")
        
        return result.mask(invalid).astype(dtype)
    
    if dtype == "boolean":
        def key(value):
            return str(value).strip().casefold()
        
        true_values = rule.get("true_values", [True, 1, "1.0", "sim"])
        false_values = rule.get("false_values", [False, 0, "0.0", "não", "nao"])
        positive, negative = {key(v) for v in true_values}, {key(v) for v in false_values}

        if positive & negative:
            raise ValueError("true_values e false_values contêm valores iguais")
        
        values = series.astype("string").str.strip().str.casefold()
        mapping = {**dict.fromkeys(positive, True), **dict.fromkeys(negative, False)}

        result = values.map(mapping)
        invalid = series.notna() & result.isna()

        if invalid.any() and errors == "raise":
            raise ValueError(f"Booleanos desconhecidos: {series[invalid].head().tolist()}")
        
        return result.astype("boolean")
    raise ValueError(f"Tipo não suportado: {dtype}")

def transform(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    rules = config.get("transformacoes") or {}
    drops = config.get("drop_cols") or []

    if not isinstance(rules, dict) or not isinstance(drops, list):
        raise ValueError("transformacoes deve ser um mapa e drop_cols deve ser uma lista")

    require_columns(df, [*rules, *drops], "Processamento")
    overlap = set(rules) & set(drops)

    if overlap:
        raise ValueError(f"Colunas configuradas para converter e remover: {sorted(overlap)}")

    result = df.copy()
    for column, rule in rules.items():
        if not isinstance(rule, dict) or "cast" not in rule:
            raise ValueError(f"Regra inválida para {column}: informe cast")
        
        try:
            result[column] = cast_column(result[column], rule)

        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"Conversão de '{column}' falhou: {exc}") from exc

    require_columns(result, ["description"], "Contagem de palavras")
    result["num_words"] = (
        result["description"]
        .astype("string")
        .fillna("")
        .str.count(r"\S+")
        .astype("Int64")
    )

    return result.drop(columns=drops)

def process_dataset(config_path: Path | None = None, *, dry_run: bool = False, root: Path = PATHS.root) -> pd.DataFrame:
    config_path = config_path or root / "src/config/sources.yaml"

    config = load_yaml(config_path)
    dataset = config["dataset"]
    source = project_path(dataset["entrada"], root)
    destination = project_path(dataset["saida"], root)

    if source == destination:
        raise ValueError("entrada e saida devem ser arquivos diferentes")
    
    if destination.suffix.lower() != ".parquet":
        raise ValueError("Use saida .parquet para preservar os tipos configurados")
    
    if source.suffix.lower() == ".csv":
        rules = config.get("transformacoes") or {}
        options = {"sep": ",", "encoding": "utf-8-sig", **(dataset.get("leitura") or {})}

        if set(options) - {"sep", "encoding", "decimal", "thousands", "na_values", "keep_default_na"}:
            raise ValueError("Opção de leitura de CSV não suportada")

        df = pd.read_csv(source, dtype={col: "string" for col in rules}, **options)
    elif source.suffix.lower() == ".parquet":
        df = pd.read_parquet(source)

    else:
        raise ValueError("A entrada deve ser .csv ou .parquet")
    
    result = transform(df, config)
    quality = config.get("qualidade") or {}
    report_path = project_path(quality["relatorio"], root) if quality.get("relatorio") and not dry_run else None

    if report_path in {source, destination}:
        raise ValueError("O relatório de qualidade não pode substituir o dataset")

    validate_quality(result, quality, root=root, report_path=report_path)
    
    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Só substitui a saída depois que concluir o write
        temporary = destination.with_suffix(".parquet.tmp")

        try:
            result.to_parquet(temporary, index=False, engine="pyarrow")
            temporary.replace(destination)

        finally:
            temporary.unlink(missing_ok=True)
            
    print(f"[Pipeline{' / dry-run' if dry_run else ''}] Processados {len(result)} linhas x {len(result.columns)} colunas.")
    return result
