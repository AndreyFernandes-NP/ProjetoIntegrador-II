"""
profiles.py
-----------
Aprende uma representação tabular com autoencoder e agrupa chamados semelhantes
com K-Means, publicando apenas um identificador categórico de perfil por chamado.
"""
from copy import deepcopy
import json
from pathlib import Path
from typing import Self

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.feature_common import FeatureStage, set_seed, device_for, save_metadata
from src.data.validator import write_json

def build_autoencoder(input_dim: int, hidden_dim: int, latent_dim: int):
    """Cria a rede de reconstrução, o encoder será usado exclusivamente para o clustering."""
    from torch import nn

    class Autoencoder(nn.Module):
        """Rede densa com camada latente linear e reconstrução das entradas normalizadas."""

        def __init__(self):
            """Inicializa encoder e decoder sem acessar dados ou rótulos."""
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim))
            self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))

        def forward(self, values):
            """Reconstrói uma matriz de entradas numéricas."""
            return self.decoder(self.encoder(values))

    return Autoencoder()

class TabularProfiles(FeatureStage):
    """Autoencoder + K-Means ajustados no treino."""

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Seleciona as features, converte números e trata categorias ausentes em uma cópia."""
        result = data[self.config["features"]].copy()

        for col in self.config["preprocessing"]["numeric"]:
            result[col] = pd.to_numeric(result[col], errors="raise").astype(float)

            if np.isinf(result[col]).any():
                raise ValueError(f"{col}: infinito nos dados tabulares")
            
        for col in self.config["preprocessing"]["categorical"]:
            result[col] = result[col].astype("string").fillna("__missing__").astype(str)

        return result

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame) -> Self:
        """Ajusta transformações, autoencoder e centróides só no treino, val escolhe a época."""
        import torch
        set_seed(self.seed)

        self.device = device_for(self.config.get("device", "auto"))
        prep = self.config["preprocessing"]

        if len(prep["numeric"] + prep["categorical"]) != len(set(prep["numeric"] + prep["categorical"])) or set(prep["numeric"] + prep["categorical"]) != set(self.config["features"]):
            raise ValueError("numeric/categorical devem cobrir exatamente as features tabulares")
        
        steps = []
        if prep["numeric"]:
            steps.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                                                ("scaler", StandardScaler())]), prep["numeric"]))
            
        if prep["categorical"]:
            steps.append(("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), prep["categorical"]))
            
        self.preprocessor = ColumnTransformer(steps)
        values = np.asarray(self.preprocessor.fit_transform(self.clean(train)), dtype="float32")
        val = np.asarray(self.preprocessor.transform(self.clean(validation)), dtype="float32")
        params, settings = self.config["network"], self.config["training"]

        self.architecture = {"input_dim": values.shape[1], "hidden_dim": params["hidden_dim"], "latent_dim": params["latent_dim"]}
        self.model = build_autoencoder(**self.architecture).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(settings["learning_rate"]))
        x, x_val = torch.from_numpy(values).to(self.device), torch.from_numpy(val).to(self.device)
        best, stale, best_state = float("inf"), 0, None

        self.history = {"loss": [], "val_loss": []}
        generator = torch.Generator().manual_seed(self.seed)
        for _ in range(settings["epochs"]):
            self.model.train()

            order = torch.randperm(len(x), generator=generator).to(self.device)
            total = 0.
            for start in range(0, len(x), settings["batch_size"]):
                batch = x[order[start:start + settings["batch_size"]]]
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(self.model(batch), batch)
                if not torch.isfinite(loss):
                    raise ValueError("Loss tabular não finita")
                
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(batch)

            self.model.eval()
            with torch.no_grad():
                validation_loss = float(torch.nn.functional.mse_loss(self.model(x_val), x_val))

            self.history["loss"].append(total / len(x))
            self.history["val_loss"].append(validation_loss)

            if validation_loss < best:
                best, stale = validation_loss, 0
                best_state = deepcopy(self.model.state_dict())
            else:
                stale += 1
                if stale >= settings["patience"]:
                    break

        if best_state is None:
            raise ValueError("Treino tabular não produziu uma época válida")
        
        self.model.load_state_dict(best_state)
        self.model.eval()

        with torch.no_grad():
            latent = self.model.encoder(x).cpu().numpy()

        clusters = self.config["clustering"]["n_clusters"]
        if len(np.unique(latent, axis=0)) < clusters:
            raise ValueError("Há menos representações distintas no treino que n_clusters")
        
        self.latent_scaler = StandardScaler().fit(latent)
        self.clusterer = KMeans(n_clusters=clusters, random_state=self.seed, n_init=20)
        self.clusterer.fit(self.latent_scaler.transform(latent))
        self.details.update(train_rows=len(train), validation_rows=len(validation),
                            n_clusters=clusters, target_used=False, best_validation_loss=best,
                            train_cluster_counts=np.bincount(self.clusterer.labels_, minlength=clusters).tolist())
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Atribui um perfil usando encoder/escala/centróides já ajustados, sem refit."""
        import torch

        values = np.asarray(self.preprocessor.transform(self.clean(data)), dtype="float32")
        self.model.eval()
        with torch.inference_mode():
            latent = self.model.encoder(torch.from_numpy(values).to(self.device)).cpu().numpy()
            
        labels = self.clusterer.predict(self.latent_scaler.transform(latent))
        return pd.DataFrame({"tabular_profile_cluster": pd.Categorical(labels, categories=range(self.config["clustering"]["n_clusters"]))}, index=data.index)

    def save(self, directory: Path) -> None:
        """Salva pesos, arquitetura e transformações para reproduzir os mesmos perfis."""
        import torch

        save_metadata(self, directory)
        torch.save(self.model.state_dict(), directory / "autoencoder.pt")
        joblib.dump((self.preprocessor, self.latent_scaler, self.clusterer), directory / "preprocessors.joblib")
        write_json(directory / "network.json", {"architecture": self.architecture, "history": self.history})

    @classmethod
    def restore(cls, directory: Path, config: dict, root: Path, seed: int = 42) -> Self:
        """Restaura o modelo salvo; IDs de cluster continuam os mesmos da execução original."""
        import torch

        instance = cls(config, root, seed)
        instance.device = device_for(config.get("device", "auto"))
        network = json.loads((directory / "network.json").read_text(encoding="utf-8"))

        instance.architecture, instance.history = network["architecture"], network["history"]
        instance.model = build_autoencoder(**instance.architecture).to(instance.device)
        instance.model.load_state_dict(torch.load(directory / "autoencoder.pt", map_location=instance.device, weights_only=True))
        instance.model.eval()
        instance.preprocessor, instance.latent_scaler, instance.clusterer = joblib.load(directory / "preprocessors.joblib")
        
        return instance
