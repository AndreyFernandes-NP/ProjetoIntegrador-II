"""
loader.py
---------
Carrega a configuração dos modelos, seleciona suas features e organiza as entradas
na divisão compartilhada de treino, validação e teste, impedindo o uso indevido do target.
Downloads são armazenados na pasta models/ da raiz do projeto.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from src.config import PATHS
from src.data.processor import load_yaml, project_path, require_columns
from src.data.splitter import DatasetSplit, get_or_create_split

_CACHE_LOCK = RLock()


@contextmanager
def _keras_models_cache(root: Path):
    """Direciona downloads internos de Keras Applications para models/keras/.

    Keras 3 fixa seu diretório durante o import e as factories não recebem
    cache_dir. O ajuste temporário também funciona se Keras já foi importado.
    O lock serializa as chamadas deste loader; não altera KERAS_HOME do processo.
    """
    from keras.src.backend import config as keras_config

    directory = (Path(root) / "models" / "keras").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with _CACHE_LOCK:
        previous = keras_config._KERAS_DIR
        keras_config._KERAS_DIR = str(directory)
        try:
            yield
        finally:
            keras_config._KERAS_DIR = previous


def download_model_file(url: str, *, root: Path = PATHS.root, sha256_hash: str | None = None) -> Path:
    """Baixa/reutiliza um arquivo .keras em models/downloads/, inclusive via código."""
    import keras
    _, url = model_source({"fonte": {"link": url}})
    directory = (Path(root) / "models").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    filename = sha256(url.encode()).hexdigest() + ".keras"
    path = keras.utils.get_file(
        fname=filename, origin=url, cache_dir=str(directory), cache_subdir="downloads",
        file_hash=sha256_hash, hash_algorithm="sha256",
    )
    return Path(path)

@dataclass
class ModelPartition:
    ids: pd.Series
    X: pd.DataFrame
    y: pd.Series | None

@dataclass
class ModelInputs:
    X: pd.DataFrame
    y: pd.Series | None
    config: dict
    model: Any = None
    partitions: dict[str, ModelPartition] = field(default_factory=dict)

def model_source(config: dict) -> tuple[str, str]:
    source = config.get("fonte") or {}
    link = source.get("link")

    if link is not None and not isinstance(link, str):
        raise ValueError("fonte.link deve ser texto ou null")
    
    if link and link.strip():
        link = link.strip()
        parsed = urlparse(link)

        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("fonte.link deve ser uma URL HTTP(S)")
        
        if not parsed.path.lower().endswith(".keras"):
            raise ValueError("fonte.link deve apontar para um modelo completo .keras")
        
        return "link", link
    
    name = source.get("nome")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Informe fonte.link ou fonte.nome para o modelo habilitado")
    
    return "nome", name.strip()

def load_model(config: dict, *, root: Path = PATHS.root):
    import keras
    kind, value = model_source(config)

    source = config["fonte"]
    if kind == "link":
        path = download_model_file(value, root=root, sha256_hash=source.get("sha256"))

        return keras.saving.load_model(str(path), compile=False, safe_mode=True)
    
    # Não permite import de módulos arbitrários ou execução de expressões do YAML
    factory = getattr(keras.applications, value, None)

    if value.startswith("_") or not callable(factory) or not value[0].isupper():
        raise ValueError(f"Modelo não encontrado em keras.applications: {value}")
    
    params = dict(source.get("parametros") or {})
    if "input_shape" in params:
        params["input_shape"] = tuple(params["input_shape"])

    with _keras_models_cache(root):
        return factory(**params)

def select_features(df: pd.DataFrame, config: dict, target: str | None) -> tuple[pd.DataFrame, pd.Series | None]:
    features = config.get("features")

    if not isinstance(features, list) or not features or not all(isinstance(c, str) for c in features):
        raise ValueError("features deve ser uma lista não vazia de nomes de colunas")
    
    if len(set(features)) != len(features):
        raise ValueError("features contém colunas repetidas")
    
    if target is not None and not isinstance(target, str):
        raise ValueError("target_col deve ser um nome de coluna ou null")
    
    if target in features:
        raise ValueError("target_col não pode estar em features (vazamento do alvo)")
    
    require_columns(df, features + ([target] if target else []), "Modelo")
    return df.loc[:, features].copy(), df[target].copy() if target else None

def prepare_models(config_path: Path | None = None, *, df: pd.DataFrame | None = None,
                   load: bool = False, only: str | None = None,
                   use_split: bool = True, split_result: DatasetSplit | None = None,
                   dry_run: bool = False, rebuild_split: bool = False,
                   root: Path = PATHS.root) -> dict[str, ModelInputs]:

    if dry_run and load:
        raise ValueError("dry_run não permite carregar/baixar modelos")

    config = load_yaml(config_path or root / "src/config/models.yaml")
    global_config = config["global_config"]
    models = config["modelos"]
    
    if not isinstance(models, dict):
        raise ValueError("modelos deve ser um mapa de modelos habilitados")
    
    if only is not None and only not in models:
        raise ValueError(f"Modelo desconhecido: {only}")
    
    if df is None:
        df = pd.read_parquet(project_path(global_config["dataset"], root))

    prepared = {}
    for name, experiment in models.items():
        if only is not None and name != only:
            continue

        if not isinstance(experiment, dict) or not isinstance(experiment.get("enabled", False), bool):
            raise ValueError(f"Configuração inválida para {name}; enabled deve ser true/false")
        
        if not experiment.get("enabled", False):
            if only is not None:
                raise ValueError(f"Modelo desabilitado: {name}; configure enabled: true")
            
            continue

        is_encoder = experiment.get("role") == "encoder"
        if isinstance(experiment.get("features"), list) and global_config.get("target_col") in experiment["features"]:
            raise ValueError("O alvo global não pode entrar nas features")
        if is_encoder and experiment.get("target_col") is not None:
            raise ValueError("Extratores não recebem target_col")
        if is_encoder:
            from src.models.outputs import validate_outputs
            validate_outputs(experiment)
        target = None if is_encoder else experiment.get("target_col", global_config.get("target_col"))
        X, y = select_features(df, experiment, target)
        model_source(experiment)
        prepared[name] = ModelInputs(X=X, y=y, config=experiment)

    # Todos os modelos usam os mesmos chamados em cada conjunto.
    if use_split:
        split_result = split_result or get_or_create_split(
            df, config, root=root, dry_run=dry_run, rebuild=rebuild_split)

        if split_result is not None:
            id_col = split_result.report["configuration"]["id_column"]

            for item in prepared.values():
                target = (None if item.config.get("role") == "encoder" else
                          item.config.get("target_col", global_config.get("target_col")))

                for name, frame in split_result.frames.items():
                    X, y = select_features(frame, item.config, target)
                    item.partitions[name] = ModelPartition(ids=frame[id_col].copy(), X=X, y=y)

    # Valida todos os modelos selecionados antes de baixar qualquer modelo.
    if load:
        for item in prepared.values():
            if item.config.get("role") == "encoder":
                # Instancia a interface; ajuste/download ocorre na pipeline de extração.
                from src.models.encoders import create_encoder
                item.model = create_encoder(item.config, global_config.get("split", {}).get("seed", 42), root)
            else:
                item.model = load_model(item.config, root=root)
            
    return prepared
