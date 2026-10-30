"""
custom.py
---------
Ponto de extensão para implementar e registrar modelos ou cálculos personalizados,
definindo como ajustar, gerar resultados, salvar e restaurar seus componentes.
"""

import numpy as np

from src.models.base import FeatureEncoder, checked_vectors

class TextRuleSignals(FeatureEncoder):
    """Exemplo opcional, baseado em regras: contagem de palavras e de dígitos.

    Não é um modelo de sentimento/urgência, o YAML quem decide quais índices publicar.
    Para adaptar: implemente fit/transform/save/load_artifacts e registre a classe,
    ajustes aprendidos ficam em fit, exclusivamente sobre X_train sem y.
    """

    def fit(self, X_train, X_validation=None) -> FeatureEncoder:
        if self.config.get("modality") != "text":
            raise ValueError("TextRuleSignals exige modality: text")
        
        self.fitted = True
        return self

    def transform(self, X):
        self.check_fitted()
        texts = X[self.config["features"]].astype("string").fillna("").agg(" ".join, axis=1)

        return checked_vectors(np.column_stack([texts.str.split().str.len(), texts.str.count(r"\d")]), len(X))

    def save(self, directory):
        directory.mkdir(parents=True, exist_ok=True)

    def load_artifacts(self, directory):
        # Modelos próprios devem restaurar aqui seus pesos/pré-processadores.
        pass

CUSTOM_ENCODERS = {"TextRuleSignals": TextRuleSignals}
