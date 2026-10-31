"""
semantic_text.py
----------------
Codifica título e descrição com um encoder semântico congelado e compara o relato
com exemplos configuráveis.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.processor import require_columns
from src.models.feature_common import device_for

def unit_vectors(values: np.ndarray) -> np.ndarray:
    """Normaliza vetores por linha, zero representa campo ausente."""
    values = np.asarray(values, dtype=np.float32)

    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Embeddings devem ser uma matriz finita")
    
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)

def combine_fields(title_vectors: np.ndarray, description_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Combina os dois campos com peso igual e retorna seu cosseno de concordância.

    Se um campo estiver ausente, o outro define o vetor e a concordância recebe
    zero como ausência de informação, sem interpretação de contradição.
    """
    title, description = unit_vectors(title_vectors), unit_vectors(description_vectors)

    if title.shape != description.shape:
        raise ValueError("Título e descrição precisam de embeddings alinhados")
    
    if np.any((np.linalg.norm(title, axis=1) == 0) & (np.linalg.norm(description, axis=1) == 0)):
        raise ValueError("Informe ao menos título ou descrição em cada chamado")
    
    return unit_vectors(title + description), np.clip(np.sum(title * description, axis=1), -1, 1)

class SemanticTextEncoder:
    """Encoder congelado com mean pooling mascarado conforme o checkpoint escolhido."""

    def __init__(self, snapshot: Path, device: str = "auto", batch_size: int = 32, max_length: int = 128):
        """Carrega pesos locais; não consulta target nem ajusta a rede."""
        from transformers import AutoModel, AutoTokenizer

        if type(batch_size) is not int or batch_size < 1 or type(max_length) is not int or max_length < 8:
            raise ValueError("batch_size/max_length inválidos")
            
        self.snapshot = Path(snapshot)
        self.device = device_for(device)
        self.batch_size, self.max_length = batch_size, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

        limit = min(self.tokenizer.model_max_length,
                    getattr(self.model.config, "max_position_embeddings", max_length))

        if max_length > limit:
            raise ValueError(f"semantic.max_length excede o limite do encoder: {limit}")

    def encode(self, texts: list[str]) -> np.ndarray:
        """Codifica textos e calcula média dos tokens excluindo padding."""
        import torch

        if not texts:
            return np.empty((0, self.model.config.hidden_size), dtype=np.float32)

        output = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                inputs = self.tokenizer(texts[start:start+self.batch_size], padding=True,
                    truncation=True, max_length=self.max_length, return_tensors="pt")

                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                hidden = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

                output.append(pooled.cpu().float().numpy())

        return unit_vectors(np.concatenate(output))

    def encode_frame(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Deduplica textos, preserva a associação às linhas e utiliza ambos os campos."""
        require_columns(data, ["title", "description"], "Encoder semântico")

        fields = data[["title", "description"]].astype("string").fillna("")
        for col in fields:
            fields[col] = fields[col].str.replace(r"\s+", " ", regex=True).str.strip()

        if (fields.to_numpy(dtype=object) == "").all(axis=1).any() or data.empty:
            raise ValueError("Informe título ou descrição para todos os chamados")

        unique = list(dict.fromkeys([v for v in fields.to_numpy().ravel() if v]))
        encoded = self.encode(unique)
        lookup = {text: encoded[i] for i, text in enumerate(unique)}
        lookup[""] = np.zeros(encoded.shape[1], dtype=np.float32)
        title, description = [np.stack([lookup[v] for v in fields[col]]) for col in fields]

        return combine_fields(title, description)

    def prototype_vectors(self, prototypes: dict[str, list[str]]) -> dict[str, np.ndarray]:
        """Codifica os exemplos de intenção sem acessar dados ou rótulos do dataset."""
        if not prototypes or any(not values or not all(isinstance(v, str) and v.strip() for v in values) for values in prototypes.values()):
            raise ValueError("Protótipos devem conter textos não vazios")

        return {name: self.encode(values) for name, values in prototypes.items()}

    def save(self, directory: Path) -> None:
        """Salva pesos/tokenizer locais para inferência sem rede ou novo download."""
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory, safe_serialization=True)
        self.tokenizer.save_pretrained(directory)

def intent_similarities(vectors: np.ndarray, prototypes: dict[str, np.ndarray]) -> pd.DataFrame:
    """Retorna o maior cosseno por grupo, sem softmax, rótulos ou limiar de rejeição.

    Valores em [-1, 1] medem proximidade aos exemplos e não são probabilidades.
    Mais de um grupo pode apresentar similaridade elevada no mesmo chamado.
    """
    if not prototypes:
        raise ValueError("Informe ao menos um grupo de exemplos semânticos")
    
    values = unit_vectors(vectors)
    result = {}

    for name, examples in prototypes.items():
        examples = unit_vectors(examples)

        if not len(examples) or examples.shape[1] != values.shape[1]:
            raise ValueError(f"Embeddings de exemplos incompatíveis: {name}")
        
        result[name] = np.clip(values @ examples.T, -1, 1).max(axis=1)
        
    return pd.DataFrame(result)
