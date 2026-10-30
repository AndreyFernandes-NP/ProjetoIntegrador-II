"""
multimodal.py
-------------
Combina as colunas geradas pelos modelos intermediários através do ID dos registros
e implementa o classificador final, incluindo pré-processamento, treinamento e persistência.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.data.validator import missing, write_json
from src.models.base import checked_vectors, fit_network, predict_batches

def feature_table(ids, outputs, id_column):
    ids = pd.Series(ids).astype("string").reset_index(drop=True)

    if missing(ids).any() or ids.duplicated().any():
        raise ValueError("IDs ausentes/duplicados nas representações")
    
    if not isinstance(outputs, pd.DataFrame) or len(outputs) != len(ids) or outputs.columns.duplicated().any():
        raise ValueError("Saídas devem ser um DataFrame nomeado com uma linha por ID")
    
    if id_column in outputs or outputs.isna().any().any():
        raise ValueError("Saídas não podem conter ID ou valores ausentes")
    
    table = outputs.reset_index(drop=True).copy()
    table.insert(0, id_column, ids)

    return table

def fuse_tables(metadata, blocks, id_column):
    result = metadata.copy()
    result[id_column] = result[id_column].astype("string")

    if result[id_column].duplicated().any() or missing(result[id_column]).any():
        raise ValueError("IDs inválidos na fusão")
    
    for name, block in blocks.items():
        if block[id_column].duplicated().any() or set(block[id_column]) != set(result[id_column]):
            raise ValueError(f"IDs de {name} não correspondem exatamente ao dataset")
        
        if (set(result.columns) & set(block.columns)) - {id_column}:
            raise ValueError("Nomes de colunas colidem durante a fusão")
        
        result = result.merge(block, on=id_column, how="left", validate="one_to_one", sort=False)

    return result

class FusionClassifier:
    def __init__(self, config, seed=42):
        self.config = config
        self.seed = seed

    def clean(self, X):
        if X.columns.tolist() != self.features:
            raise ValueError("Schema/ordem de features difere do treino multimodal")
        
        result = X.copy()
        for column in self.numeric:
            result[column] = pd.to_numeric(result[column], errors="raise").astype(float)

            if np.isinf(result[column]).any():
                raise ValueError(f"{column}: infinito nas features do multimodal")
            
        for column in self.categorical:
            result[column] = result[column].astype("string").fillna("__missing__").astype(str)

        return result

    def fit(self, X_train, y_train, X_val, y_val):
        import keras
        keras.utils.set_random_seed(self.seed)

        if X_train.columns.tolist() != X_val.columns.tolist():
            raise ValueError("Schema diferente entre treino e validação")
        
        self.features = X_train.columns.tolist()
        self.categorical = self.config.get("categorical_features", [])

        if not set(self.categorical).issubset(self.features):
            raise ValueError("categorical_features deve ser subconjunto das features finais")
        
        self.numeric = [column for column in self.features if column not in self.categorical]
        transformers = []

        if self.numeric:
            transformers.append(("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler())]), self.numeric))
            
        if self.categorical:
            transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical))

        self.preprocessor = ColumnTransformer(transformers)
        train = checked_vectors(self.preprocessor.fit_transform(self.clean(X_train)), len(X_train))
        val = checked_vectors(self.preprocessor.transform(self.clean(X_val)), len(X_val))
        self.input_dimensions = train.shape[1]
        self.labels = LabelEncoder().fit(np.asarray(y_train).astype(str))

        if len(self.labels.classes_) < 2:
            raise ValueError("Classificação final exige pelo menos duas classes no treino")
        
        y = self.labels.transform(np.asarray(y_train).astype(str))
        validation_y = self.labels.transform(np.asarray(y_val).astype(str))
        inputs = keras.Input(shape=(self.input_dimensions,))
        hidden = inputs

        for units in self.config.get("hidden_dims", [64, 32]):
            hidden = keras.layers.Dense(int(units), activation="relu")(hidden)
            hidden = keras.layers.Dropout(float(self.config.get("dropout", 0.2)))(hidden)

        output = keras.layers.Dense(len(self.labels.classes_), activation="softmax")(hidden)
        self.model = keras.Model(inputs, output)
        training = self.config.get("training", {})
        self.model.compile(optimizer=keras.optimizers.Adam(training.get("learning_rate", 0.001)),
                           loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.history = fit_network(self.model, train, y, val, validation_y, training, self.seed)

        return self

    def predict_proba(self, X):
        values = checked_vectors(self.preprocessor.transform(self.clean(X)), len(X))

        return predict_batches(self.model, values)

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save(directory / "model.keras")
        joblib.dump(self.preprocessor, directory / "preprocessor.joblib")
        write_json(directory / "metadata.json", {"format_version": 2, "features": self.features,
                   "input_dimensions": self.input_dimensions,
                   "classes": self.labels.classes_.tolist(), "config": self.config, "seed": self.seed,
                   "history": self.history})

    @classmethod
    def restore(cls, directory):
        import keras
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))

        if metadata.get("format_version") != 2:
            raise ValueError("Classificador antigo: gere uma nova execução")
        
        result = cls(metadata["config"], metadata["seed"])
        result.features = metadata["features"]
        result.labels = LabelEncoder()
        result.labels.classes_ = np.array(metadata["classes"])
        result.categorical = result.config.get("categorical_features", [])
        result.numeric = [c for c in result.features if c not in result.categorical]
        result.input_dimensions = metadata["input_dimensions"]
        result.preprocessor = joblib.load(directory / "preprocessor.joblib")
        result.model = keras.saving.load_model(directory / "model.keras", compile=False, safe_mode=True)
        
        return result
