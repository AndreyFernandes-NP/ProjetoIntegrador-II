"""
validator.py
------------
Lógica de validação e inspeção de qualidade do dataset, com relatório em JSON.
"""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import tempfile

import pandas as pd

from src.config import PATHS

class ValidationError(ValueError):
    """Relatório disponível mesmo quando a qualidade interrompe a execução."""
    def __init__(self, report: dict):
        self.report = report
        super().__init__(f"Qualidade reprovada: {report['errors']} erro(s). Consulte o relatório.")

def write_json(path: Path, data: dict) -> None:
    """Publica um JSON completo, caso uma escrita seja interrompida preserva a versão anterior."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")

        temporary.replace(path)

    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

def missing(series: pd.Series) -> pd.Series:
    """String vazia ou só de espaços também conta como ausente."""
    return series.isna() | series.astype("string").str.strip().eq("").fillna(False)

def image_file(value: str, base: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()

def inspect_images(series: pd.Series, base: Path) -> tuple[dict, dict]:
    """Abre cada caminho uma vez, verifica decodificação e calcula a hash dos bytes."""
    from PIL import Image

    valid, invalid = {}, {}
    for value in sorted(series[~missing(series)].astype(str).unique()):
        path = image_file(value, base)
        try:
            with Image.open(path) as picture:
                picture.verify()

            with Image.open(path) as picture:
                picture.load()
                size = list(picture.size)

            valid[value] = {"sha256": sha256(path.read_bytes()).hexdigest(), "size": size}

        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
            invalid[value] = f"{type(exc).__name__}: {exc}"

    return valid, invalid

def validate_quality(df: pd.DataFrame, config: dict | None = None, *,
                     root: Path = PATHS.root, report_path: Path | None = None,
                     fail_on_error: bool = True) -> dict:
    """Executa regras, salva relatório opcional e então lança um ValidationError."""
    cfg = config or {}

    if not isinstance(cfg, dict):
        raise ValueError("qualidade deve ser um mapa YAML")
    
    known = {"relatorio", "cols_obrigatorias", "cols_nao_nulas", "cols_unicas",
             "cols_numericas", "valores_permitidos", "intervalos", "limite_nulos",
             "textos", "imagens"}
    
    if set(cfg) - known:
        raise ValueError(f"Regras de qualidade desconhecidas: {sorted(set(cfg) - known)}")
    
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "rows": len(df),
              "columns": len(df.columns), "issues": [], "profile": {}}

    def issue(code, detail, count=1, severity="error"):
        report["issues"].append({"code": code, "severity": severity,
                                 "count": int(count), "detail": detail})

    def columns(value, context):
        if not isinstance(value, list) or not all(isinstance(c, str) for c in value):
            raise ValueError(f"{context} deve ser uma lista de colunas")
        
        available = []
        for col in value:
            if col not in df.columns:
                issue("missing_column", f"{context}: coluna '{col}' ausente")

            else:
                available.append(col)

        return available

    if df.empty:
        issue("empty_dataset", "Dataset vazio")

    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicate_columns:
        issue("duplicate_columns", f"Nomes de coluna repetidos: {duplicate_columns}")

    else:
        report["profile"] = {
            str(col): {"dtype": str(df[col].dtype), "missing": int(missing(df[col]).sum()),
                       "missing_fraction": float(missing(df[col]).mean()) if len(df) else 0.0,
                       "unique_non_null": int(df.loc[~missing(df[col]), col].nunique())}
            for col in df.columns
        }

        duplicates = int(df.duplicated().sum())
        if duplicates:
            issue("duplicate_rows", "Linhas inteiramente repetidas", duplicates, "warning")
            
        columns(cfg.get("cols_obrigatorias", []), "cols_obrigatorias")
        for key in ("cols_nao_nulas", "cols_unicas", "cols_numericas"):
            for col in columns(cfg.get(key, []), key):
                values = df[col]
                if key == "cols_nao_nulas" and missing(values).any():
                    issue("null_values", f"'{col}' contém valores ausentes", int(missing(values).sum()))

                elif key == "cols_unicas":
                    count = values[~missing(values)].duplicated().sum()

                    if count:
                        issue("duplicate_key", f"'{col}' contém valores repetidos", count)

                elif key == "cols_numericas":
                    if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
                        issue("not_numeric", f"'{col}' não é numérica")

        for col, allowed in (cfg.get("valores_permitidos") or {}).items():
            if not isinstance(allowed, list) or not allowed:
                raise ValueError(f"valores_permitidos.{col} deve ser uma lista não vazia")
            
            if columns([col], "valores_permitidos"):
                bad = ~missing(df[col]) & ~df[col].isin(allowed)

                if bad.any():
                    issue("invalid_category", f"Valores fora do domínio em '{col}'", bad.sum())

        for col, limits in (cfg.get("intervalos") or {}).items():
            if not isinstance(limits, dict) or not limits or set(limits) - {"min", "max"}:
                raise ValueError(f"intervalos.{col} deve conter min e/ou max")
            
            if "min" in limits and "max" in limits and limits["min"] > limits["max"]:
                raise ValueError(f"Intervalo invertido em '{col}'")
            
            if columns([col], "intervalos"):
                values = pd.to_numeric(df[col], errors="coerce")
                bad = (~missing(df[col]) & values.isna()) | values.isin([float("inf"), -float("inf")])

                if "min" in limits:
                    bad |= values.lt(limits["min"])
                if "max" in limits:
                    bad |= values.gt(limits["max"])
                if bad.any():
                    issue("out_of_range", f"Valores inválidos ou fora do intervalo em '{col}'", bad.sum())

        null_cfg = cfg.get("limite_nulos")
        if null_cfg is not None:
            threshold = null_cfg["fracao"]

            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
                raise ValueError("limite_nulos.fracao deve estar entre 0 e 1")
            
            severity = null_cfg.get("severidade", "warning")
            if severity not in {"warning", "error"}:
                raise ValueError("severidade deve ser warning ou error")
            
            selected = columns(null_cfg.get("colunas", list(df.columns)), "limite_nulos")
            for col in selected:
                fraction = report["profile"][col]["missing_fraction"]
                if fraction > threshold:
                    issue("high_missing_fraction", f"'{col}': {fraction:.2%} ausentes (limite {threshold:.2%})",
                          report["profile"][col]["missing"], severity)

        text_cfg = cfg.get("textos") or {}
        target = text_cfg.get("target_col")
        text_stats = {}
        target_exists = bool(columns([target], "textos.target_col")) if target else False

        for col in columns(text_cfg.get("colunas", []), "textos.colunas"):
            subset = df.loc[~missing(df[col])]
            text_stats[col] = {"unique": int(subset[col].nunique()),
                               "duplicate_rows": int(subset[col].duplicated().sum())}
            
            if target_exists:
                conflicts = int((subset.groupby(col)[target].nunique() > 1).sum())
                text_stats[col]["values_with_multiple_targets"] = conflicts

                if conflicts:
                    issue("ambiguous_text", f"Textos idênticos em '{col}' associados a múltiplos alvos",
                          conflicts, "warning")
                    
        report["texts"] = text_stats

        image_cfg = cfg.get("imagens") or {}
        if image_cfg:
            col = image_cfg["coluna"]
            if columns([col], "imagens.coluna"):
                base = Path(image_cfg.get("base_dir", "data"))
                base = base if base.is_absolute() else root / base
                valid, invalid = inspect_images(df[col], base)
                report["images"] = {"unique_paths": len(valid) + len(invalid),
                                    "unique_contents": len({v["sha256"] for v in valid.values()}),
                                    "invalid_paths": invalid, "missing_rows": int(missing(df[col]).sum())}
                
                if invalid:
                    issue("invalid_image", "Imagens inexistentes ou ilegíveis; detalhes em images.invalid_paths",
                          len(invalid))
                    
                flag = image_cfg.get("attachment_col")
                if flag and columns([flag], "imagens.attachment_col"):
                    values = df[flag]
                    flag_invalid = ~missing(values) & ~values.isin([True, False, 0, 1])
                    if flag_invalid.any():
                        issue("invalid_attachment_flag", f"'{flag}' deve ser booleano ou 0/1", flag_invalid.sum())

                    inconsistency = (values.eq(True).fillna(False) & missing(df[col])) | (
                        values.eq(False).fillna(False) & ~missing(df[col]))
                    
                    if inconsistency.any():
                        issue("attachment_mismatch", f"'{flag}' diverge da presença de '{col}'; verificar placeholders",
                              inconsistency.sum(), "warning")

    report["errors"] = sum(item["severity"] == "error" for item in report["issues"])
    report["warnings"] = sum(item["severity"] == "warning" for item in report["issues"])
    report["passed"] = report["errors"] == 0

    if report_path is not None:
        write_json(report_path, report)

    print(f"[Qualidade] {report['errors']} erro(s), {report['warnings']} aviso(s)")

    for item in report["issues"]:
        print(f"  [{item['severity']}] {item['detail']} — {item['count']}")

    if fail_on_error and not report["passed"]:
        raise ValidationError(report)
    
    return report
