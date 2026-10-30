"""
fusion.py
---------
Combina textos, sinais numéricos e categorias em uma representação interna única
e aprende o target com regressão logística/MLP.
"""
import json
from pathlib import Path
from typing import Self
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.processor import require_columns
from src.data.validator import write_json
from src.models.feature_common import device_for, set_seed

def feature_columns(config: dict) -> list[str]:
    """Retorna a lista explícita de entradas, mantendo a ordem definida no YAML."""
    features = config["features"]

    return features["numeric"] + features["categorical"] + [c for block in features["text"].values() for c in block]

def clean_features(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Seleciona apenas as entradas autorizadas e prepara blocos textuais sem duplicação."""
    require_columns(data, feature_columns(config), "Fusão")

    features = config["features"]
    result = data[features["numeric"] + features["categorical"]].copy()

    for col in features["numeric"]:
        result[col] = pd.to_numeric(result[col], errors="raise").astype(float)

        if np.isinf(result[col]).any():
            raise ValueError(f"{col}: valores infinitos")
        
    for col in features["categorical"]:
        result[col] = result[col].astype("string").fillna("__missing__").astype(str)
        
    for name, columns in features["text"].items():
        def combine(row) -> str:
            """Preserva conteúdo e negações, removendo apenas campos idênticos na linha."""
            seen, values = set(), []

            for value in row:
                text = "" if pd.isna(value) else " ".join(str(value).split())

                if text and text.casefold() not in seen:
                    seen.add(text.casefold())
                    values.append(text)

            return "\n".join(values) or "__missingtext__"
        
        result[f"__text_{name}"] = data[columns].apply(combine, axis=1)

    return result

def build_preprocessor(config: dict, params: dict) -> ColumnTransformer:
    """Cria transformações ajustáveis exclusivamente no treino e concatena seus blocos."""
    features = config["features"]

    transformers = []
    if features["numeric"]:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler())]), features["numeric"]))
        
    if features["categorical"]:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True), features["categorical"]))

    for name in features["text"]:
        vectorizer = TfidfVectorizer(max_features=params["text_max_features"],
            ngram_range=(1, params["text_ngram_max"]), min_df=params["text_min_df"],
            lowercase=True, sublinear_tf=True, strip_accents="unicode", dtype=np.float32)
        
        transformers.append((f"text_{name}", vectorizer, f"__text_{name}"))

    return ColumnTransformer(transformers, sparse_threshold=1.0)

def classification_metrics(y, probabilities, n_classes: int) -> dict:
    """Calcula métricas multiclasse, incluindo classes ausentes nas predições."""
    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.shape != (len(y), n_classes) or not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Probabilidades de classificação inválidas")
    
    # Corrige apenas arredondamento float32 antes da tolerância estrita do log_loss.
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

    predicted = probabilities.argmax(axis=1)
    return {"accuracy": float(accuracy_score(y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "f1_macro": float(f1_score(y, predicted, labels=range(n_classes), average="macro", zero_division=0)),
            "log_loss": float(log_loss(y, probabilities, labels=range(n_classes)))}

def build_mlp(input_dim: int, hidden_dims: list[int], dropout: float, n_classes: int):
    """Constrói uma MLP com logits de saída na ordem fixa das classes do YAML."""
    import torch.nn as nn

    layers = []
    previous = input_dim

    for width in hidden_dims:
        layers.extend([nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)])
        previous = width

    layers.append(nn.Linear(previous, n_classes))

    return nn.Sequential(*layers)

class EarlyFusionClassifier:
    """Classificador único sobre features combinadas; não ajusta os geradores anteriores."""

    def __init__(self, config: dict, params: dict, seed: int = 42):
        """Registra o contrato das entradas e os hiperparâmetros de uma tentativa."""
        self.config, self.params, self.seed = config, params, seed
        self.classes = np.asarray(config["classes"], dtype=str)
        self.history = []
        self.details = {}

    def encode_labels(self, values) -> np.ndarray:
        """Codifica rótulos conhecidos; rejeita nulos e classes fora da lista autorizada."""
        series = pd.Series(values)

        if series.isna().any():
            raise ValueError("Target contém valores ausentes")
        
        mapping = {label: i for i, label in enumerate(self.classes)}
        result = series.astype(str).map(mapping)

        if result.isna().any():
            raise ValueError(f"Targets desconhecidos: {series[result.isna()].unique().tolist()}")
        
        return result.to_numpy(dtype=int)

    def fit(self, train: pd.DataFrame, y_train, validation: pd.DataFrame, y_validation) -> Self:
        """Ajusta pré-processamento no treino; usa validação para seleção de época da MLP."""
        y, val_y = self.encode_labels(y_train), self.encode_labels(y_validation)

        if set(y) != set(range(len(self.classes))):
            raise ValueError("O treino deve conter todas as classes configuradas")
        
        self.preprocessor = build_preprocessor(self.config, self.params)
        x = sparse.csr_matrix(self.preprocessor.fit_transform(clean_features(train, self.config)), dtype=np.float32)
        val_x = sparse.csr_matrix(self.preprocessor.transform(clean_features(validation, self.config)), dtype=np.float32)

        shape = x.shape
        if shape is None or len(shape) < 2:
            raise ValueError("Representação da fusão sem dimensões válidas")
        
        self.input_dimensions = shape[1]
        
        if not self.input_dimensions or not np.isfinite(x.data).all() or not np.isfinite(val_x.data).all():
            raise ValueError("Representação da fusão vazia ou não finita")
        
        if self.params["kind"] == "logistic":
            self.model = LogisticRegression(C=self.params["C"], class_weight=self.params["class_weight"],
                max_iter=self.config["training"]["logistic_max_iter"], solver="lbfgs", random_state=self.seed)
            
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                self.model.fit(x, y)

            self.details["warnings"] = [str(w.message) for w in caught]
            self.details["iterations"] = self.model.n_iter_.tolist()

        else:
            self._fit_mlp(x, y, val_x, val_y)
            
        return self

    def _fit_mlp(self, x, y, val_x, val_y) -> None:
        """Treina em lotes, com AdamW, pesos de classes opcionais e early stopping externo."""
        import torch
        set_seed(self.seed)

        self.device = device_for(self.config.get("device", "auto"))
        self.model = build_mlp(self.input_dimensions, self.params["hidden_dims"], self.params["dropout"], len(self.classes)).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params["learning_rate"], weight_decay=self.params["weight_decay"])

        weights = None
        if self.params["class_weight"] == "balanced":
            weights = torch.tensor(len(y) / (len(self.classes) * np.bincount(y, minlength=len(self.classes))), dtype=torch.float32, device=self.device)

        loss_function = torch.nn.CrossEntropyLoss(weight=weights)
        settings = self.config["training"]
        metric = self.config["search"]["metric"]
        best_score, best_state, stale = -np.inf, None, 0
        rng = np.random.default_rng(self.seed)
        for epoch in range(settings["epochs"]):
            self.model.train()
            order, total = rng.permutation(len(y)), 0.

            for start in range(0, len(y), self.params["batch_size"]):
                indices = order[start:start + self.params["batch_size"]]
                batch = torch.from_numpy(x[indices].toarray()).to(self.device)
                labels = torch.as_tensor(y[indices], dtype=torch.long, device=self.device)
                optimizer.zero_grad()
                loss = loss_function(self.model(batch), labels)

                if not torch.isfinite(loss):
                    raise ValueError("Loss multimodal não finita")
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                total += float(loss.detach()) * len(indices)

            probabilities = self._matrix_proba(val_x)
            metrics = classification_metrics(val_y, probabilities, len(self.classes))
            self.history.append({"epoch": epoch + 1, "train_loss": total / len(y), **{f"validation_{k}": v for k, v in metrics.items()}})

            if metrics[metric] > best_score + settings["min_delta"]:
                best_score, stale = metrics[metric], 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                self.details["best_epoch"] = epoch + 1
            else:
                stale += 1
                if stale >= settings["patience"]:
                    break
        if best_state is None:
            raise ValueError("MLP não produziu uma época válida")
        
        self.model.load_state_dict(best_state)
        self.model.eval()

    def _matrix_proba(self, values) -> np.ndarray:
        """Infere matrizes prontas em lotes para limitar a conversão de esparso para denso."""
        if isinstance(self.model, LogisticRegression):
            return self.model.predict_proba(values)
        
        import torch
        self.model.eval()

        output = []
        with torch.inference_mode():
            for start in range(0, values.shape[0], self.params["batch_size"]):
                tensor = torch.from_numpy(values[start:start + self.params["batch_size"]].toarray()).to(self.device)
                output.append(torch.softmax(self.model(tensor), dim=1).cpu().numpy())

        return np.concatenate(output)

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Retorna probabilidades na ordem de classes persistida, sem refit."""
        if data.empty:
            raise ValueError("Nenhum chamado para classificar")
        
        values = sparse.csr_matrix(self.preprocessor.transform(clean_features(data, self.config)), dtype=np.float32)
        if not np.isfinite(values.data).all():
            raise ValueError("Entradas de inferência não finitas")
        
        return self._matrix_proba(values)

    def save(self, directory: Path) -> None:
        """Salva pré-processamento, classificador, classes, configuração e histórico."""
        directory.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.preprocessor, directory / "preprocessor.joblib")
        if isinstance(self.model, LogisticRegression):
            joblib.dump(self.model, directory / "classifier.joblib")

        else:
            import torch
            
            torch.save(self.model.state_dict(), directory / "classifier.pt")
        write_json(directory / "metadata.json", {"format_version": 4, "config": self.config, "params": self.params,
                   "seed": self.seed, "input_dimensions": self.input_dimensions, "history": self.history, "details": self.details})

    @classmethod
    def restore(cls, directory: Path) -> Self:
        """Restaura artefatos locais confiáveis sem ajustar vocabulário, escalas ou pesos."""
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))

        if metadata.get("format_version") != 4:
            raise ValueError("Formato de classificador incompatível")
        
        instance = cls(metadata["config"], metadata["params"], metadata["seed"])
        instance.input_dimensions = metadata["input_dimensions"]
        instance.history, instance.details = metadata["history"], metadata["details"]
        instance.preprocessor = joblib.load(directory / "preprocessor.joblib")

        if instance.params["kind"] == "logistic":
            instance.model = joblib.load(directory / "classifier.joblib")

        else:
            import torch

            instance.device = device_for(instance.config.get("device", "auto"))
            instance.model = build_mlp(instance.input_dimensions, instance.params["hidden_dims"], instance.params["dropout"], len(instance.classes)).to(instance.device)
            instance.model.load_state_dict(torch.load(directory / "classifier.pt", map_location=instance.device, weights_only=True))
            instance.model.eval()

        return instance
        