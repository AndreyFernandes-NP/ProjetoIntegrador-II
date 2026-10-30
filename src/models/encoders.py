"""
encoders.py
-----------
Implementa os modelos intermediários de dados tabulares, visão computacional e PLN,
gerando resultados sem utilizar o target final e permitindo salvar e restaurar os modelos.
"""

import json
from pathlib import Path
from typing import Any
from typing import Self

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize
from sklearn.metrics.pairwise import cosine_similarity

from src.config import PATHS
from src.data.validator import missing, write_json
from src.models.base import FeatureEncoder, checked_vectors, fit_network, predict_batches
from src.models.custom import CUSTOM_ENCODERS

def linked(config):
    value = config.get("fonte", {}).get("link")
    return isinstance(value, str) and bool(value.strip())

def keras_encoder(config, root=PATHS.root):
    import keras
    from src.models.loader import load_model

    model: Any = load_model(config, root=root)
    layer = config.get("output_layer")
    
    if layer:
        model = keras.Model(model.inputs, model.get_layer(layer).output)

    model.trainable = False
    return model

class TabularEncoder(FeatureEncoder):
    def clean(self, X):
        features = self.config["features"]
        if isinstance(features, str):
            features = [features]
        result = X.loc[:, features].copy()

        for col in self.numeric:
            result[col] = pd.to_numeric(result[col], errors="raise").astype(float)

            if np.isinf(result[col]).any():
                raise ValueError(f"'{col}' contém infinito")
            
        for col in self.categorical:
            result[col] = result[col].astype("string").fillna("__missing__").astype(str)

        return result

    def fit(self, X_train, X_validation=None) -> Self:
        import keras

        keras.utils.set_random_seed(self.seed)

        self.numeric = list(self.config["preprocessing"].get("numeric", []))
        self.categorical = list(self.config["preprocessing"].get("categorical", []))
        declared = self.numeric + self.categorical
        features = self.config["features"]
        if isinstance(features, str):
            features = [features]

        if len(declared) != len(set(declared)) or set(declared) != set(features):
            raise ValueError("numeric e categorical devem cobrir exatamente as features, sem repetições")
        
        transformers = []
        if self.numeric:
            transformers.append(("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler())]), self.numeric))
            
        if self.categorical:
            transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical))
            
        self.preprocessor = ColumnTransformer(transformers)
        train = self.preprocessor.fit_transform(self.clean(X_train))

        toarray = getattr(train, "toarray", None)
        train = toarray() if callable(toarray) else train
            
        train = np.asarray(train, dtype=np.float32)
        if linked(self.config):
            self.model = keras_encoder(self.config, getattr(self, "root", PATHS.root))
        else:
            if self.config["fonte"].get("nome") != "TabularAutoencoder":
                raise ValueError("Deep learning tabular: use TabularAutoencoder ou link de encoder .keras")
            
            if X_validation is None:
                raise ValueError("O autoencoder exige entradas de validação separadas")
            
            params = self.config["fonte"].get("parametros", {})
            dim, hidden = int(params.get("latent_dim", 8)), int(params.get("hidden_dim", 32))

            if dim < 1 or hidden < 1:
                raise ValueError("latent_dim e hidden_dim devem ser positivos")
            
            inputs = keras.Input(shape=(train.shape[1],))
            encoded = keras.layers.Dense(hidden, activation="relu")(inputs)
            latent = keras.layers.Dense(dim, activation="linear", name="embedding")(encoded)
            decoded = keras.layers.Dense(hidden, activation="relu")(latent)
            decoded = keras.layers.Dense(train.shape[1])(decoded)
            autoencoder = keras.Model(inputs, decoded)
            settings = self.config.get("training", {})

            autoencoder.compile(optimizer=keras.optimizers.Adam(settings.get("learning_rate", 0.001)), loss="mse")
            val = self.preprocessor.transform(self.clean(X_validation))

            toarray = getattr(val, "toarray", None)
            val = toarray() if callable(toarray) else val
            val = np.asarray(val, dtype=np.float32)

            # O objetivo é reconstruir X, nunca prever target_category.
            self.history = fit_network(autoencoder, train, train, val, val, settings, self.seed)
            representation = self.config.get("representation", "reconstruction_error")

            if representation not in {"reconstruction_error", "latent"}:
                raise ValueError("representation deve ser reconstruction_error ou latent")
            
            self.model = autoencoder if representation == "reconstruction_error" else keras.Model(inputs, latent)

        self.fitted = True
        return self

    def transform(self, X):
        self.check_fitted()

        data = self.preprocessor.transform(self.clean(X))
        toarray = getattr(data, "toarray", None)
        data = toarray() if callable(toarray) else data
        data = np.asarray(data, dtype="float32")

        values = predict_batches(self.model, data, int(self.config.get("batch_size", 64)))
        if not linked(self.config) and self.config.get("representation", "reconstruction_error") == "reconstruction_error":
            # Erro de reconstrução no espaço pré-processado; não é probabilidade.
            return checked_vectors(np.mean(np.square(values - data), axis=1, keepdims=True), len(X))
        
        return values

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump((self.preprocessor, self.numeric, self.categorical), directory / "preprocessor.joblib")

        self.model.save(directory / "encoder.keras")

class VisionEncoder(FeatureEncoder):
    def __init__(self, config, seed=42, root=PATHS.root):
        super().__init__(config, seed)
        self.root = root
        self.cache = {}

    def fit(self, X_train, X_validation=None) -> Self:
        import keras
        keras.utils.set_random_seed(self.seed)

        if len(self.config["features"]) != 1:
            raise ValueError("Visão espera uma única coluna de caminhos de imagem")
        
        self.model = keras_encoder(self.config, getattr(self, "root", PATHS.root))
        preprocessing = self.config["preprocessing"]
        self.size = tuple(preprocessing["image_size"])

        if len(self.size) != 2 or any(type(v) is not int or v < 1 for v in self.size):
            raise ValueError("image_size deve conter altura e largura positivas")
        
        if preprocessing.get("function") not in {"mobilenet_v2", "resnet", "resnet_v2", "efficientnet", "none"}:
            raise ValueError("Pré-processamento visual não suportado")
        
        self.fitted = True
        return self

    def transform(self, X):
        import keras
        from PIL import Image, ImageOps

        self.check_fitted()
        column = self.config["features"][0]

        if missing(X[column]).any():
            raise ValueError("Imagem ausente: esta versão exige evidência visual em todas as linhas")
        
        base = Path(self.config["preprocessing"].get("base_dir", "data"))
        base = base if base.is_absolute() else self.root / base
        paths = [(Path(v) if Path(v).is_absolute() else base / v).resolve() for v in X[column].astype(str)]
        pending = list(dict.fromkeys(path for path in paths if path not in self.cache))
        batch_size = int(self.config.get("batch_size", 32))
        if batch_size < 1:
            raise ValueError("batch_size deve ser positivo")
        
        functions = {"mobilenet_v2": keras.applications.mobilenet_v2.preprocess_input,
                     "resnet": keras.applications.resnet.preprocess_input,
                     "resnet_v2": keras.applications.resnet_v2.preprocess_input,
                     "efficientnet": keras.applications.efficientnet.preprocess_input,
                     "none": lambda x: x}
        
        for start in range(0, len(pending), batch_size):
            batch_paths = pending[start:start + batch_size]
            pixels = []

            for path in batch_paths:
                with Image.open(path) as image:
                    rgb = ImageOps.exif_transpose(image).convert("RGB")
                    rgb = rgb.resize((self.size[1], self.size[0]), Image.Resampling.BILINEAR)
                    pixels.append(np.asarray(rgb, dtype="float32"))

            vectors = predict_batches(self.model, functions[self.config["preprocessing"]["function"]](np.stack(pixels)), batch_size)
            self.cache.update(zip(batch_paths, vectors))

        return checked_vectors(np.stack([self.cache[path] for path in paths]), len(X))

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save(directory / "encoder.keras")

class TextEncoder(FeatureEncoder):
    def texts(self, X):
        # Mantém números/códigos de erro e a ordem dos campos declarados.
        return X[self.config["features"]].astype("string").fillna("").agg(" \n ".join, axis=1).tolist()

    def fit(self, X_train, X_validation=None) -> Self:
        if linked(self.config):
            self.model = keras_encoder(self.config, getattr(self, "root", PATHS.root))

        else:
            method = self.config["fonte"].get("nome")
            if method not in {"TfidfSVD", "TfidfSimilarity"}:
                raise ValueError("PLN: use TfidfSimilarity, TfidfSVD ou link .keras com tokenização incorporada")
            
            params = self.config["fonte"].get("parametros", {})
            self.vectorizer = TfidfVectorizer(max_features=int(params.get("max_features", 10000)),
                                             ngram_range=tuple(params.get("ngram_range", [1, 2])),
                                             min_df=params.get("min_df", 1), sublinear_tf=True)
            
            matrix = self.vectorizer.fit_transform(self.texts(X_train))
            if method == "TfidfSimilarity":
                references = params.get("references")
                if not isinstance(references, list) or not references or not all(isinstance(s, str) and s.strip() for s in references):
                    raise ValueError("TfidfSimilarity exige references: lista ordenada de textos")
                
                self.references = self.vectorizer.transform(references)
                if (self.references.getnnz(axis=1) == 0).any():
                    raise ValueError("Referência textual sem termos no vocabulário de treino; revise references")
                
                self.fitted = True
                return self
            
            dim = min(int(params.get("dimensions", 64)), matrix.shape[0] - 1, matrix.shape[1] - 1)

            if dim < 1:
                raise ValueError("PLN exige pelo menos dois exemplos e dois termos no treino")
            
            self.svd = TruncatedSVD(n_components=dim, random_state=self.seed)
            self.svd.fit(matrix)

        self.fitted = True
        return self

    def transform(self, X):
        self.check_fitted()
        texts = self.texts(X)

        if linked(self.config):
            import tensorflow as tf
            # Contrato do arquivo remoto: batch de strings -> matriz de embeddings.
            return predict_batches(self.model, tf.constant(texts), int(self.config.get("batch_size", 64)))
        
        if self.config["fonte"]["nome"] == "TfidfSimilarity":
            matrix = self.vectorizer.transform(texts)

            return checked_vectors(np.clip(cosine_similarity(matrix, self.references), 0, 1), len(X))
        
        return checked_vectors(normalize(self.svd.transform(self.vectorizer.transform(texts))), len(X))

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        if linked(self.config):
            self.model.save(directory / "encoder.keras")

        else:
            projection = self.references if self.config["fonte"]["nome"] == "TfidfSimilarity" else self.svd
            joblib.dump((self.vectorizer, projection), directory / "preprocessor.joblib")

ENCODERS = {"tabular": TabularEncoder, "vision": VisionEncoder, "text": TextEncoder}

def create_encoder(config, seed=42, root=PATHS.root):
    source_name = config.get("fonte", {}).get("nome")
    if not linked(config) and source_name in CUSTOM_ENCODERS:
        encoder = CUSTOM_ENCODERS[source_name](config, seed)
        setattr(encoder, "root", root)
        return encoder
    
    modality = config.get("modality")
    if modality not in ENCODERS:
        raise ValueError(f"Modalidade inválida: {modality}; use tabular, vision ou text")
    
    args = {"root": root} if modality == "vision" else {}
    encoder = ENCODERS[modality](config, seed, **args)
    setattr(encoder, "root", root)
    return encoder

def save_encoder(encoder, directory):
    encoder.save(directory)
    write_json(directory / "metadata.json", {"format_version": 2, "config": encoder.config, "seed": encoder.seed, "history": encoder.history})

def restore_encoder(directory, root=PATHS.root):
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))

    if metadata.get("format_version") != 2:
        raise ValueError("Extrator antigo: gere uma nova execução com outputs explícitos")
    
    encoder = create_encoder(metadata["config"], metadata["seed"], root)
    if not linked(encoder.config) and encoder.config["fonte"]["nome"] in CUSTOM_ENCODERS:
        encoder.load_artifacts(directory)
        encoder.fitted = True

        return encoder
    
    modality = encoder.config["modality"]
    if (directory / "encoder.keras").exists():
        import keras

        model: Any = keras.saving.load_model(directory / "encoder.keras", compile=False, safe_mode=True)
        model.trainable = False
        setattr(encoder, "model", model)
        
    if modality == "tabular":
        preprocessor, numeric, categorical = joblib.load(directory / "preprocessor.joblib")

        setattr(encoder, "preprocessor", preprocessor)
        setattr(encoder, "numeric", numeric)
        setattr(encoder, "categorical", categorical)

    elif modality == "text" and not linked(encoder.config):
        vectorizer, projection = joblib.load(directory / "preprocessor.joblib")
        setattr(encoder, "vectorizer", vectorizer)

        if encoder.config["fonte"]["nome"] == "TfidfSimilarity":
            setattr(encoder, "references", projection)
        else:
            setattr(encoder, "svd", projection)

    elif modality == "vision":
        setattr(encoder, "size", tuple(encoder.config["preprocessing"]["image_size"]))

    encoder.fitted = True
    return encoder
