"""
outputs.py
----------
Valida as saídas configuradas no models.yaml e converte os resultados dos modelos
em colunas nomeadas, aplicando seleção de valores, categorias, limites e tipos.
"""

import re

import numpy as np
import pandas as pd

from src.models.base import checked_vectors

def validate_outputs(config):
    outputs = config.get("outputs")

    if not isinstance(outputs, list) or not outputs:
        raise ValueError("Declare outputs: uma lista não vazia de colunas; expansão automática foi removida")
    
    names = []
    for spec in outputs:
        if not isinstance(spec, dict):
            raise ValueError("Cada output deve ser um mapa")
        
        name, operation = spec.get("name"), spec.get("operation")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError("outputs.name deve usar letras minúsculas, números e underscore")
        
        if operation not in {"value", "argmax", "max", "threshold"}:
            raise ValueError(f"Operação de saída inválida: {operation}")
        
        dtype = spec.get("dtype", "float")
        expected = {"value": {"float", "int"}, "max": {"float"},
                    "argmax": {"category"}, "threshold": {"bool"}}
        
        if dtype not in expected[operation]:
            raise ValueError(f"dtype incompatível com {operation}: use {sorted(expected[operation])}")
        
        if operation in {"value", "threshold"}:
            if type(spec.get("index")) is not int or spec["index"] < 0:
                raise ValueError("value/threshold exige index inteiro >= 0")
            
        if operation == "threshold":
            value = spec.get("threshold")
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value):
                raise ValueError("threshold exige um limiar numérico finito")
            
        if "labels" in spec:
            labels = spec["labels"]
            if operation != "argmax" or not isinstance(labels, list) or not labels or not all(isinstance(v, str) and v for v in labels) or len(set(labels)) != len(labels):
                raise ValueError("labels deve mapear, sem repetições, cada índice de argmax para uma categoria")
            
        if "range" in spec:
            bounds = spec["range"]
            if dtype == "category" or not isinstance(bounds, list) or len(bounds) != 2 or not all(type(v) in {int, float} and np.isfinite(v) for v in bounds) or bounds[0] > bounds[1]:
                raise ValueError("range deve ser [mínimo, máximo] numérico e finito")
            
        names.append(name)

    if len(set(names)) != len(names):
        raise ValueError("Nomes de outputs duplicados")
    
    return names

def output_types(config, names):
    return {spec["name"]: spec.get("dtype", "float") for name in names for spec in config["modelos"][name]["outputs"]}

def project_outputs(values, config, rows):
    """Seleciona valores/reduções; nunca copia todas as dimensões implicitamente."""
    validate_outputs(config)
    values = checked_vectors(values, rows)
    result = {}

    for spec in config["outputs"]:
        name, operation = spec["name"], spec["operation"]
        dtype = spec.get("dtype", "float")

        if operation in {"value", "threshold"}:
            if spec["index"] >= values.shape[1]:
                raise ValueError(f"{name}: index fora da saída do modelo {values.shape}")
            
            column = values[:, spec["index"]]
            if operation == "threshold":
                column = column >= spec["threshold"]

        elif operation == "max":
            column = values.max(axis=1)

        else:
            labels = spec.get("labels", [f"class_{i}" for i in range(values.shape[1])])

            if len(labels) != values.shape[1]:
                raise ValueError(f"{name}: labels deve ter {values.shape[1]} categorias, na ordem do modelo")
            
            column = np.asarray(labels)[values.argmax(axis=1)]
        if dtype == "int" and not np.equal(column, np.floor(column)).all():
            raise ValueError(f"{name}: valor fracionário não pode ser convertido para int")
        
        if "range" in spec:
            low, high = spec["range"]
            if ((column < low) | (column > high)).any():
                raise ValueError(f"{name}: valores fora do range declarado {spec['range']}")
            
        result[name] = pd.Series(column, dtype={"float": "float32", "int": "int64", "bool": "bool", "category": "string"}[dtype])
        
    return pd.DataFrame(result)
