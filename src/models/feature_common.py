"""
feature_common.py
-----------------
Contratos e utilitários da geração de colunas sem o target final, incluindo controle
aleatório, seleção de dispositivo, persistência e download de modelos para models/.
"""
from abc import ABC, abstractmethod
from hashlib import sha256
import os
from pathlib import Path
import random
from typing import Self
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from src.data.processor import project_path
from src.data.validator import write_json

class FeatureStage(ABC):
    """Uma etapa recebe apenas suas features e retorna colunas nomeadas no mesmo índice."""

    def __init__(self, config: dict, root: Path, seed: int = 42):
        """Guarda configuração, raiz e seed; não carrega redes durante a construção."""
        self.config, self.root, self.seed = config, Path(root), seed
        self.details = {}

    @abstractmethod
    def fit(self, train: pd.DataFrame, validation: pd.DataFrame) -> Self:
        """Ajusta somente com treino; validação pode selecionar a melhor época."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Retorna somente os outputs declarados, com o índice e a ordem de data."""
        raise NotImplementedError

    @abstractmethod
    def save(self, directory: Path) -> None:
        """Persiste os parâmetros necessários à geração futura das mesmas colunas."""
        raise NotImplementedError

def set_seed(seed: int) -> None:
    """Fixa RNGs Python/NumPy/Torch e pede kernels determinísticos quando disponíveis.

    Não garante identidade bit a bit entre versões de bibliotecas ou dispositivos.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

def device_for(name: str = "auto"):
    """Resolve cpu/cuda/auto; erro explícito se CUDA foi solicitado sem estar disponível."""
    import torch

    if name not in {"auto", "cpu", "cuda"}:
        raise ValueError("device deve ser auto, cpu ou cuda")
    
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA solicitado, mas indisponível")
    
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)

def hf_source(source: dict) -> tuple[str, str]:
    """Resolve nome ou URL de repositório Hugging Face; link preenchido tem prioridade.

    URLs apontam ao repositório completo (modelo + tokenizer/processor), não a um
    arquivo .keras isolado. Não executa código remoto do repositório.
    """
    revision = source.get("revision", "main")
    value = source.get("link")

    if value:
        parsed = urlparse(value)
        parts = parsed.path.strip("/").split("/")

        if parsed.scheme != "https" or parsed.netloc != "huggingface.co" or parsed.query or parsed.fragment or len(parts) != 2:
            raise ValueError("fonte.link deve ser https://huggingface.co/autor/repositorio")
        
        repo = "/".join(parts)

    else:
        repo = source.get("nome", "")

    if not isinstance(repo, str) or len(repo.split("/")) != 2 or any(p in {"", ".", ".."} for p in repo.split("/")):
        raise ValueError("fonte.nome deve ser autor/repositorio")
    
    if not isinstance(revision, str) or not revision:
        raise ValueError("fonte.revision deve ser uma revisão ou commit")
    
    return repo, revision

def download_pretrained(source: dict, root: Path) -> Path:
    """Baixa snapshot para models/huggingface e reaproveita o cache por commit.

    Prefere safetensors, baixa pesos PyTorch binários somente se não houver essa
    alternativa. Dependências de imagem/texto são baixadas junto dos pesos.
    """
    from huggingface_hub import HfApi, snapshot_download

    repo, revision = hf_source(source)
    cache = Path(root) / "models" / "huggingface"
    cache.mkdir(parents=True, exist_ok=True)

    offline = source.get("local_files_only", False)
    if offline:
        path = Path(snapshot_download(repo, revision=revision, cache_dir=str(cache), local_files_only=True))

        if not (list(path.glob("*.safetensors")) or list(path.glob("pytorch_model*.bin"))):
            raise ValueError("Snapshot offline não contém pesos completos")
        
        return path
    
    info = HfApi().model_info(repo, revision=revision)
    files = [item.rfilename for item in (info.siblings or [])]
    safe = any("/" not in f and f.endswith(".safetensors") for f in files)
    weights = ["*.safetensors", "*.safetensors.index.json"] if safe else ["pytorch_model*.bin", "pytorch_model*.bin.index.json"]

    return Path(snapshot_download(repo, revision=info.sha, cache_dir=str(cache),
                                  allow_patterns=["*.json", "*.txt", "*.model"] + weights,
                                  ignore_patterns=["onnx/*", "openvino/*", "tf_model*", "flax_model*"]))

def image_records(data: pd.DataFrame, column: str, base: Path) -> pd.DataFrame:
    """Resolve caminhos e hashes de imagens; rejeita ausências e preserva o índice."""
    rows, cache = [], {}

    for value in data[column]:
        if pd.isna(value) or not str(value).strip():
            raise ValueError("image_path ausente: forneça uma imagem válida")
        
        path = project_path(str(value), base)
        if path not in cache:
            cache[path] = sha256(path.read_bytes()).hexdigest()

        rows.append({"path": str(path), "sha256": cache[path]})

    return pd.DataFrame(rows, index=data.index)

def save_metadata(stage: FeatureStage, directory: Path) -> None:
    """Grava configuração, seed e detalhes da etapa junto de seus artefatos."""
    directory.mkdir(parents=True, exist_ok=True)
    
    write_json(directory / "metadata.json", {"schema_version": 3, "config": stage.config,
                                              "seed": stage.seed, "details": stage.details})
