"""
base.py
-------
Define a interface dos modelos intermediários e funções auxiliares para treinamento,
inferência em lotes e validação das saídas numéricas.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd

class FeatureEncoder(ABC):
    def __init__(self, config: dict, seed: int = 42):
        self.config = config
        self.seed = seed
        self.history = {}
        self.fitted = False

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, X_validation: pd.DataFrame | None = None) -> Self:
        """Ajusta usando somente entradas do treino; validação é opcional."""

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Uma linha de vetores para cada linha de X, na mesma ordem."""

    @abstractmethod
    def save(self, directory: Path) -> None:
        pass

    def check_fitted(self):
        if not self.fitted:
            raise ValueError("Extrator ainda não ajustado/carregado")

def checked_vectors(values, rows: int) -> np.ndarray:
    values = np.asarray(values, dtype="float32")

    if values.ndim != 2 or values.shape[0] != rows or values.shape[1] == 0:
        raise ValueError(f"Extrator deve retornar matriz [linhas, dimensões]; recebido {values.shape}")
    
    if not np.isfinite(values).all():
        raise ValueError("Representação contém NaN ou infinito")
    
    return values

def predict_batches(model, values, batch_size=64):
    import keras
    outputs = []

    for start in range(0, len(values), batch_size):
        result = model(values[start:start + batch_size], training=False)

        if isinstance(result, (dict, list, tuple)):
            raise ValueError("O encoder deve expor uma única saída vetorial")
        
        outputs.append(keras.ops.convert_to_numpy(result))

    if not outputs:
        raise ValueError("Não é possível transformar um conjunto vazio")
    
    return checked_vectors(np.concatenate(outputs), len(values))

def fit_network(model, X, y, X_val, y_val, settings: dict, seed: int):
    """Keras com backend TensorFlow, batches e validação explícita (sem test)."""
    import keras
    import tensorflow as tf

    epochs, batch = int(settings.get("epochs", 30)), int(settings.get("batch_size", 64))

    if epochs < 1 or batch < 1:
        raise ValueError("epochs e batch_size devem ser positivos")
    
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    train = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(len(X), seed=seed).batch(batch).with_options(options)
    validation = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch).with_options(options)
    history = model.fit(train, validation_data=validation, epochs=epochs, verbose=0, shuffle=False,
                        callbacks=[keras.callbacks.EarlyStopping(
                            monitor="val_loss", patience=int(settings.get("patience", 5)),
                            restore_best_weights=True)])
    
    return {key: [float(v) for v in values] for key, values in history.history.items()}
