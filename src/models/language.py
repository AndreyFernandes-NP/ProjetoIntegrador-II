"""
language.py
-----------
Recalcula sentimento por NLI e gera similaridades semânticas de título e descrição.
Mantém o adaptador de tópicos anterior para restaurar execuções legadas.
"""
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd

from src.models.feature_common import FeatureStage, set_seed, device_for, download_pretrained, save_metadata

def topic_text(data: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Combina campos textuais sem repetir conteúdo idêntico na mesma linha.

    Normaliza espaços para comparação; preserva a grafia original do primeiro
    campo encontrado e não remove frases distintas ou negações.
    """
    def combine(row) -> str:
        """Une apenas os trechos não vazios e distintos de uma linha."""
        seen, parts = set(), []
        for value in row:
            if pd.isna(value):
                continue
            value = " ".join(str(value).split())
            key = value.casefold()
            if key and key not in seen:
                seen.add(key)
                parts.append(value)
        return "\n".join(parts)
    return data[columns].apply(combine, axis=1)

def validate_topic_settings(settings: dict) -> None:
    """Valida a rejeição configurável e a fonte opcional do modelo de tópicos."""
    from src.models.feature_common import hf_source
    mode = settings.get("scoring", "relative")
    if mode not in {"relative", "entailment"}:
        raise ValueError("topics.scoring deve ser relative ou entailment")
    if "fonte" in settings:
        if mode != "entailment":
            raise ValueError("topics.fonte exige topics.scoring: entailment")
        hf_source(settings["fonte"])
    reject = settings.get("rejection", {})
    if reject and mode != "entailment":
        raise ValueError("topics.rejection exige scoring: entailment")
    if mode == "entailment":
        for key in ("min_entailment", "min_margin"):
            value = reject.get(key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"topics.rejection.{key} deve estar entre 0 e 1")
        fallback = reject.get("label")
        labels = [c["label"] for c in settings["candidates"]]
        if not isinstance(fallback, str) or not fallback.strip() or fallback in labels:
            raise ValueError("O rótulo de rejeição deve ser não vazio e distinto dos candidatos")
    for key, minimum in (("batch_size", 1), ("max_length", 16)):
        if key in settings and (type(settings[key]) is not int or settings[key] < minimum):
            raise ValueError(f"topics.{key} deve ser inteiro >= {minimum}")

def decide_topics(support, candidates: list[dict], rejection: dict) -> pd.DataFrame:
    """Seleciona tópicos com evidência NLI independente e permite abstenção.

    Cada coluna contém p(entailment) no softmax de três classes do par
    texto/hipótese, não um softmax entre tópicos. Valores não são calibrados.
    A margem continua sendo top1-top2, agora sobre esses scores independentes.
    """
    values = np.asarray(support, dtype=float)
    labels = [c["label"] for c in candidates]
    if values.ndim != 2 or values.shape[1] != len(labels) or len(labels) < 2 or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Suportes NLI inválidos")
    order = np.argsort(-values, axis=1, kind="stable")
    rows = np.arange(len(values))
    best, second = values[rows, order[:, 0]], values[rows, order[:, 1]]
    margin = best - second
    weak = best < rejection["min_entailment"]
    ambiguous = (margin < rejection["min_margin"]) | np.isclose(margin, 0, atol=1e-12, rtol=0)
    winner = np.asarray(labels, dtype=object)[order[:, 0]]
    chosen = np.where(weak | ambiguous, rejection["label"], winner)
    reason = np.select([weak & ambiguous, weak, ambiguous],
                       ["low_evidence_and_ambiguous", "low_evidence", "ambiguous"], default="accepted")
    return pd.DataFrame({"main_topic": chosen, "semantic_margin": margin,
                         "candidate_top1": winner, "candidate_top2": np.asarray(labels)[order[:, 1]],
                         "top1_entailment": best, "top2_entailment": second,
                         "topic_decision": reason})


def weighted_score(probabilities, values) -> np.ndarray:
    """Calcula a média ponderada de valores definidos no YAML."""
    p = np.asarray(probabilities, dtype=float)
    values = np.asarray(values, dtype=float)

    if values.ndim != 1 or not np.isfinite(values).all() or p.ndim != 2 or p.shape[1] != len(values) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("Probabilidades inválidas ou pesos incompatíveis")
    
    return p @ values

def topic_decision(probabilities, labels) -> tuple[np.ndarray, np.ndarray]:
    """Retorna tópico vencedor e p(top1)-p(top2)."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != len(labels) or len(labels) < 2 or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("Tópicos exigem ao menos dois candidatos e uma matriz finita compatível")
    
    ranked = np.sort(p, axis=1)
    return np.asarray(labels)[p.argmax(axis=1)], ranked[:, -1] - ranked[:, -2]

class ContextualNLP(FeatureStage):
    """Mantém a rede base para sentimento/urgência e permite outra fonte para tópicos."""

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame) -> Self:
        """Carrega o modelo pré-treinado; não ajusta pesos nem usa rótulos dos chamados."""
        validate_topic_settings(self.config["topics"])
        set_seed(self.seed)

        self.device = device_for(self.config.get("device", "auto"))
        path = download_pretrained(self.config["fonte"], self.root)
        self._load(path)
        self.prepare_topics()
        self.details.update(pretrained_snapshot=str(path), pretrained_commit=path.name,
                            fitted_on_target=False, tasks=["sentiment", "topic", "urgency"])
        
        return self

    def _load_nli(self, path: Path, explicit=None):
        """Restaura uma rede local e identifica a posição de entailment por nome."""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True, trust_remote_code=False).to(self.device)
        model.eval()
        model.requires_grad_(False)
        labels = model.config.id2label
        matches = [int(i) for i, label in labels.items() if str(label).lower() == "entailment"]
        if explicit is None and len(matches) != 1:
            raise ValueError("Modelo não identifica entailment; configure entailment_index corretamente")
        entailment_index = matches[0] if explicit is None else int(explicit)
        if not 0 <= entailment_index < model.config.num_labels:
            raise ValueError("entailment_index inválido")
        return tokenizer, model, entailment_index

    def _load(self, path: Path) -> None:
        """Carrega a rede base de sentimento/urgência e de tópicos nas configurações antigas."""
        self.tokenizer, self.model, self.entailment_index = self._load_nli(path, self.config.get("entailment_index"))

    def prepare_topics(self, saved_path: Path | None = None) -> Self:
        """Prepara somente o classificador de tópicos, permitindo revisão sem carregar visão.

        saved_path restaura pesos locais; caso contrário, baixa/reutiliza o cache.
        Ausência de topics.fonte reutiliza a rede base já carregada.
        """
        settings = self.config["topics"]
        validate_topic_settings(settings)
        if "fonte" in settings:
            set_seed(self.seed)
            self.device = device_for(self.config.get("device", "auto"))
            path = saved_path or download_pretrained(settings["fonte"], self.root)
            self.topic_tokenizer, self.topic_model, self.topic_entailment_index = self._load_nli(path, settings.get("entailment_index"))
            self.details.update(topic_snapshot=str(path), topic_scoring=settings.get("scoring"))
        return self

    def topic_support(self, texts: list[str], hypotheses: list[str]) -> np.ndarray:
        """Calcula p(entailment) por hipótese, incluindo neutral/contradiction no denominador."""
        import torch
        settings = self.config["topics"]
        model = getattr(self, "topic_model", None)
        if model is None:
            model, tokenizer, index = self.model, self.tokenizer, self.entailment_index
        else:
            tokenizer, index = self.topic_tokenizer, self.topic_entailment_index
        batch = settings.get("batch_size", self.config.get("batch_size", 16))
        length = settings.get("max_length", self.config.get("max_length", 256))
        count = len(hypotheses)
        scores = []
        # Materializa só um lote de pares por vez.
        with torch.inference_mode():
            for start in range(0, len(texts) * count, batch):
                indices = range(start, min(start + batch, len(texts) * count))
                pairs = [(texts[i // count], hypotheses[i % count]) for i in indices]
                inputs = tokenizer([p[0] for p in pairs], [p[1] for p in pairs], padding=True,
                                   truncation="only_first", max_length=length, return_tensors="pt")
                logits = model(**{k: v.to(self.device) for k, v in inputs.items()}).logits
                scores.extend(torch.softmax(logits.float(), dim=-1)[:, index].cpu().tolist())
        return np.asarray(scores).reshape(len(texts), count)

    def classify_topics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Retorna tópico/margem e diagnóstico por linha sem ler o target ou o tópico antigo."""
        if data.empty:
            raise ValueError("Nenhum texto para classificar")
        settings = self.config["topics"]
        validate_topic_settings(settings)
        texts = topic_text(data, self.config["features"])
        if texts.eq("").any():
            raise ValueError("Informe título ou descrição não vazios")
        unique = texts.drop_duplicates().tolist()
        candidates = settings["candidates"]
        hypotheses = [c["hypothesis"] for c in candidates]
        if settings.get("scoring", "relative") == "relative":
            main, margin = topic_decision(self.probabilities(unique, hypotheses), [c["label"] for c in candidates])
            table = pd.DataFrame({"main_topic": main, "semantic_margin": margin})
        else:
            table = decide_topics(self.topic_support(unique, hypotheses), candidates, settings["rejection"])
        table.index = unique
        result = table.loc[texts.tolist()].copy()
        result.index = data.index
        return result

    def probabilities(self, texts: list[str], hypotheses: list[str]) -> np.ndarray:
        """Classifica candidatos pelo softmax dos logits de entailment, em lotes limitados.

        O texto é truncado para caber no modelo, todas as hipóteses são preservadas.
        As probabilidades são relativas aos candidatos definidos.
        """
        import torch

        count = len(hypotheses)
        batch = int(self.config.get("batch_size", 16))
        length = int(self.config.get("max_length", 256))

        if batch < 1 or length < 16 or not count:
            raise ValueError("batch_size/max_length/candidatos inválidos")
        
        logits = []
        # Evita materializar todos os tensores de todos os candidatos na memória.
        pairs = [(text, hypothesis) for text in texts for hypothesis in hypotheses]
        with torch.inference_mode():
            for start in range(0, len(pairs), batch):
                piece = pairs[start:start + batch]
                inputs = self.tokenizer([p[0] for p in piece], [p[1] for p in piece],
                                        padding=True, truncation="only_first", max_length=length, return_tensors="pt")
                
                outputs = self.model(**{key: value.to(self.device) for key, value in inputs.items()})
                logits.extend(outputs.logits[:, self.entailment_index].cpu().tolist())

        values = np.asarray(logits, dtype=float).reshape(len(texts), count)
        values -= values.max(axis=1, keepdims=True)
        exponent = np.exp(values)

        return exponent / exponent.sum(axis=1, keepdims=True)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Gera todas as quatro saídas, urgência usa o sentimento atualizado do df final e o tópico recém-identificado."""
        columns = self.config["features"]

        texts = data[columns].astype("string").fillna("").agg("\n".join, axis=1)
        if texts.str.strip().eq("").any():
            raise ValueError("Título e descrição não podem estar ambos vazios")
        
        # Repetições textuais recebem a mesma inferência e não consomem novos lotes.
        unique = texts.drop_duplicates().tolist()
        sentiment = self.config["sentiment"]["candidates"]
        sent = weighted_score(self.probabilities(unique, [x["hypothesis"] for x in sentiment]), [x["value"] for x in sentiment])

        topic_rows = data.loc[~texts.duplicated(), columns]
        diagnostics = self.classify_topics(topic_rows)
        main, margin = diagnostics["main_topic"].to_numpy(), diagnostics["semantic_margin"].to_numpy()

        diagnostics.index = unique
        self.topic_audit = diagnostics.loc[texts.tolist()].copy()
        self.topic_audit.index = data.index

        # Limita o texto original para reservar espaço aos novos sinais na entrada da urgência.
        limit = int(self.config["urgency"].get("context_characters", 700))
        context = [f"Sentimento estimado: {s:.3f}. Assunto: {t}. Chamado: {text[:limit]}"
                   for text, s, t in zip(unique, sent, main)]
        
        urgency = self.config["urgency"]["candidates"]
        urgency_prob = self.probabilities(context, [x["hypothesis"] for x in urgency])

        unit = weighted_score(urgency_prob, [x["value"] for x in urgency])
        low, high = self.config["urgency"]["range"]
        scores = low + unit * (high - low)
        table = pd.DataFrame({"sentiment_score": sent, "main_topic": main,
                              "semantic_margin": margin, "urgency_score": scores}, index=unique)
        
        result = table.loc[texts.tolist()].copy()
        result.index = data.index
        
        return result

    def save(self, directory: Path) -> None:
        """Salva rede/tokenizer da execução para reuso sem depender do estado futuro do Hub."""
        save_metadata(self, directory)
        self.model.save_pretrained(directory / "model", safe_serialization=True)
        self.tokenizer.save_pretrained(directory / "model")

        if hasattr(self, "topic_model"):
            self.topic_model.save_pretrained(directory / "topic_model", safe_serialization=True)
            self.topic_tokenizer.save_pretrained(directory / "topic_model")

    @classmethod
    def restore(cls, directory: Path, config: dict, root: Path, seed: int = 42) -> Self:
        """Carrega artefatos locais sem reajustar a rede ou consultar o Hub."""
        instance = cls(config, root, seed)
        instance.device = device_for(config.get("device", "auto"))
        instance._load(directory / "model")
        instance.prepare_topics(directory / "topic_model" if "fonte" in config["topics"] else None)
        return instance


def validate_semantic_settings(config: dict) -> None:
    """Valida exemplos, entradas e limites do PLN sem carregar ou baixar modelos."""
    from src.models.feature_common import hf_source

    if config.get("features") != ["title", "description"]:
        raise ValueError("PLN semântico exige features: [title, description], nessa ordem")
    if "topics" in config or "urgency" in config:
        raise ValueError("semantic_nlp substitui topics/urgency; remova essas configurações")
    settings = config["semantic"]
    hf_source(settings["fonte"])
    for section, minimum in ((config, 16), (settings, 8)):
        for key, lower in (("batch_size", 1), ("max_length", minimum)):
            if type(section.get(key)) is not int or section[key] < lower:
                raise ValueError(f"PLN: {key} deve ser inteiro >= {lower}")
    prototypes = settings.get("prototypes")
    if not isinstance(prototypes, dict) or not prototypes:
        raise ValueError("semantic.prototypes deve mapear colunas a listas de exemplos")
    for name, examples in prototypes.items():
        if not isinstance(name, str) or not name.isidentifier() or not name.startswith("text_"):
            raise ValueError("Colunas semânticas devem ser identificadores iniciados por text_")
        if (not isinstance(examples, list) or not examples
                or not all(isinstance(v, str) and v.strip() for v in examples)):
            raise ValueError(f"Informe exemplos textuais não vazios para {name}")
        if len(set(examples)) != len(examples):
            raise ValueError(f"Exemplos repetidos em {name}")
    candidates = config["sentiment"]["candidates"]
    hypotheses = [c["hypothesis"] for c in candidates]
    if (len(candidates) < 2 or not all(isinstance(h, str) and h.strip() for h in hypotheses)
            or len(set(hypotheses)) != len(hypotheses)):
        raise ValueError("Candidatos de sentimento devem ser distintos e não vazios")
    values = np.asarray([c["value"] for c in candidates], dtype=float)
    if not np.isfinite(values).all() or (values < -1).any() or (values > 1).any():
        raise ValueError("Valores de sentimento devem estar em [-1, 1]")


class SemanticNLP(ContextualNLP):
    """Combina o sentimento existente com similaridades independentes, sem prever target."""

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame) -> Self:
        """Carrega redes congeladas e codifica exemplos do YAML sem ajustar aos chamados.

        As partições não são usadas para aprender rótulos ou alterar protótipos.
        A assinatura mantém o contrato comum das três etapas geradoras.
        """
        from src.models.semantic_text import SemanticTextEncoder

        validate_semantic_settings(self.config)
        set_seed(self.seed)
        self.device = device_for(self.config.get("device", "auto"))
        sentiment_path = download_pretrained(self.config["fonte"], self.root)
        self._load(sentiment_path)
        settings = self.config["semantic"]
        semantic_path = download_pretrained(settings["fonte"], self.root)
        self.encoder = SemanticTextEncoder(semantic_path, str(self.device),
                                          settings["batch_size"], settings["max_length"])
        self.prototypes = self.encoder.prototype_vectors(settings["prototypes"])
        self.details.update(pretrained_snapshot=str(sentiment_path),
                            pretrained_commit=sentiment_path.name,
                            semantic_snapshot=str(semantic_path), semantic_commit=semantic_path.name,
                            fitted_on_target=False, tasks=["sentiment", "semantic_similarities"],
                            semantic_columns=list(self.prototypes),
                            pooling="masked_mean", field_combination="normalized_equal_weight",
                            similarity="max_cosine_per_prototype_group", score_range=[-1, 1])
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Substitui sentimento e gera apenas as similaridades declaradas no YAML.

        Preserva o cálculo anterior do sentimento. O encoder semântico normaliza
        espaços, deduplica textos e combina título/descrição com pesos iguais.
        """
        from src.data.processor import require_columns
        from src.models.semantic_text import intent_similarities

        require_columns(data, self.config["features"], "PLN semântico")
        if data.empty:
            raise ValueError("Nenhum chamado para enriquecer")
        texts = data[self.config["features"]].astype("string").fillna("").agg("\n".join, axis=1)
        if texts.str.strip().eq("").any():
            raise ValueError("Título e descrição não podem estar ambos vazios")
        unique = texts.drop_duplicates().tolist()
        candidates = self.config["sentiment"]["candidates"]
        scores = weighted_score(self.probabilities(unique, [c["hypothesis"] for c in candidates]),
                                [c["value"] for c in candidates])
        sentiment = pd.Series(scores, index=unique)
        vectors, _ = self.encoder.encode_frame(data[self.config["features"]])
        result = intent_similarities(vectors, self.prototypes)
        result.index = data.index
        result.insert(0, "sentiment_score", sentiment.loc[texts.tolist()].to_numpy())
        return result[[s["name"] for s in self.config["outputs"]]]

    def save(self, directory: Path) -> None:
        """Persiste ambas as redes e os exemplos codificados dentro de models/runs/."""
        save_metadata(self, directory)
        self.model.save_pretrained(directory / "model", safe_serialization=True)
        self.tokenizer.save_pretrained(directory / "model")
        self.encoder.save(directory / "semantic_model")
        np.savez_compressed(directory / "semantic_prototypes.npz", **self.prototypes)

    @classmethod
    def restore(cls, directory: Path, config: dict, root: Path, seed: int = 42) -> Self:
        """Restaura redes e protótipos salvos, sem downloads, fit ou leitura do target."""
        from src.models.semantic_text import SemanticTextEncoder, unit_vectors

        validate_semantic_settings(config)
        instance = cls(config, root, seed)
        instance.device = device_for(config.get("device", "auto"))
        instance._load(directory / "model")
        settings = config["semantic"]
        instance.encoder = SemanticTextEncoder(directory / "semantic_model", str(instance.device),
                                              settings["batch_size"], settings["max_length"])
        with np.load(directory / "semantic_prototypes.npz", allow_pickle=False) as archive:
            if set(archive.files) != set(settings["prototypes"]):
                raise ValueError("Protótipos salvos não correspondem à configuração da execução")
            instance.prototypes = {name: unit_vectors(archive[name]) for name in settings["prototypes"]}
        for name, vectors in instance.prototypes.items():
            if vectors.shape != (len(settings["prototypes"][name]), instance.encoder.model.config.hidden_size):
                raise ValueError(f"Dimensões incompatíveis nos protótipos de {name}")
        return instance
