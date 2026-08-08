"""
cleaner.py
----------
Lógica de limpeza genérica aplicada a qualquer DataFrame da pipeline.

Obs: Adicionar novas funções de limpeza se precisar ao longo do projeto.
"""

import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import PATHS

RAW_DIR     = PATHS.data_raw
CLEAN_DIR   = PATHS.data_clean

encodings = ["utf-8", "utf-8-sig", "latin1", "iso-8859-1", "cp1252"]
na_values = ['nan', '?', 'null', 'none']
separators = [";", ","]

def has_replacement_char(df: pd.DataFrame) -> bool:
    return any("�" in str(col) for col in df.columns)

def careful_load_csv(path: Path, sep: str | None = None) -> pd.DataFrame:
    errors = []

    for encoding in encodings:
        for separator in separators:
            try:
                df = pd.read_csv(path, sep=separator, encoding=encoding, low_memory=False, na_values=na_values)

                if len(df.columns) <= 1:
                    errors.append(f"encoding={encoding}, sep={repr(separator)} → apenas {len(df.columns)} coluna(s)")
                    continue

                if has_replacement_char(df):
                    errors.append(f"encoding={encoding}, sep={repr(separator)} → caractere inválido nas colunas: {list(df.columns)}")
                    continue

                #print(f"[load] '{path.name}' lido com encoding={encoding}, sep={repr(separator)}")
                return df

            except UnicodeDecodeError as e:
                errors.append(f"encoding={encoding}, sep={repr(separator)} → UnicodeDecodeError: {e}")

            except pd.errors.ParserError as e:
                errors.append(f"encoding={encoding}, sep={repr(separator)} → ParserError: {e}")

            except Exception as e:
                errors.append(f"encoding={encoding}, sep={repr(separator)} → {type(e).__name__}: {e}")

    raise ValueError(f"[Cleaner] Não foi possível ler '{path.name}' sem corromper colunas.\n"+ "\n".join(errors[-20:]))

def iter_raw_files(raw_dir):
    if not raw_dir.exists():
        return []

    return sorted(
        [path for path in raw_dir.iterdir() if path.is_file() and path.suffix.lower() == ".csv"]
    )

def purge_unicode_text(texto: object) -> str:
    """
    Remove acentos de valores de células.
    
    Sempre retorna string.
    """
    if texto is None:
        return ""

    return "".join(c for c in unicodedata.normalize("NFKD", str(texto)) if not unicodedata.combining(c))

def normalize_column_name(nome: object) -> str:
    """
    Normaliza nomes de colunas e nomes vindos do YAML.
    """
    nome_normalizado = purge_unicode_text(nome)

    nome_normalizado = (nome_normalizado.replace("\ufeff", "").strip().lower())

    nome_normalizado = re.sub(r"^\d+\s*", "", nome_normalizado)
    nome_normalizado = re.sub(r"\s*/\s*", " / ", nome_normalizado)
    nome_normalizado = re.sub(r"\s+", " ", nome_normalizado)

    return nome_normalizado.strip()

def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip de espaços em colunas de texto."""

    cols_str = df.select_dtypes(include=["object", "string"]).columns
    df[cols_str] = (df[cols_str].apply(lambda col: col.str.strip()).replace(r"^\s*$", pd.NA, regex=True))

    return df

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(how="all")
    df = df.drop_duplicates()

    df.columns = [normalize_column_name(column) for column in df.columns]
    df = clean_strings(df)

    for column in df.columns:
        series = df[column]
        dtype_name = str(series.dtype)

        if "string" in dtype_name or "object" in dtype_name:
            converted = False

            if not series.empty:
                cleaned_series = (
                    series.astype(str)
                    .str.replace(".", "", regex=False)
                    .str.replace(",", ".", regex=False)
                    .str.replace(" ", "", regex=False)
                )

                numeric_values = pd.to_numeric(cleaned_series, errors="coerce")

                if numeric_values.notna().sum() / len(series) >= 0.9:
                    df[column] = numeric_values
                    converted = True

            if not converted:
                try:
                    parsed_dates = pd.to_datetime(series, errors="coerce")
                except Exception:
                    parsed_dates = pd.Series([pd.NaT] * len(series), index=series.index)

                if parsed_dates.notna().sum() / max(1, series.notna().sum()) >= 0.8:
                    df[column] = parsed_dates

    return df

def get_null_amount(df: pd.DataFrame, nome: str) -> pd.DataFrame:
    """Imprime um relatório de colunas com valores nulos (não remove)."""
    nulos = df.isnull().sum()
    nulos = nulos[nulos > 0]

    if not nulos.empty:
        pct = (nulos / len(df) * 100).round(1)
        print(f"[nulos] '{nome}' — colunas com valores ausentes:") # debug
        for col in nulos.index:
            print(f"- {col}: {nulos[col]} ({pct[col]}%)") # debug

    return df

def main():
    files = iter_raw_files(RAW_DIR)
    if not files:
        print(f"[Cleaner] Nenhum arquivo encontrado em: {RAW_DIR}")
        return

    for file_path in files:
        try:
            df = careful_load_csv(file_path)
            cleaned_df = clean_dataframe(df)
            cleaned_df = get_null_amount(cleaned_df, file_path.name)

            output_path = CLEAN_DIR / f"{file_path.stem}-clean.csv"
            cleaned_df.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"[Cleaner] Arquivo limpo: {file_path.name} -> {output_path}")
        except Exception as exc:
            print(f"[Erro] Não foi possível processar {file_path.name}: {exc}")

if __name__ == "__main__":
    main()
