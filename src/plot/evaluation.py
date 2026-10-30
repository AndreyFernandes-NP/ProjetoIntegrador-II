"""
evaluation.py
-------------
Gráficos de avaliação do classificador final, incluindo matriz de confusão e curvas
ROC e Precision-Recall, com ordem explícita de classes e validação dos resultados.
"""
from collections.abc import Sequence, Mapping

from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

from .base import PlotResult, create_axes, finish_plot, defaults
from .relationships import plot_heatmap

def plot_confusion_matrix(y_true=None, y_pred=None, *, matrix=None, labels: Sequence | None = None,
                          normalize: str | None = None, cmap: str = "Blues",
                          ax: Axes | None = None, figsize: tuple = (8, 6), **options) -> PlotResult:
    """Exibe reais nas linhas e previstos nas colunas.

    Args:
        y_true, y_pred: Vetores alinhados. Use ambos OU matrix, nunca as duas formas.
        matrix: Matriz de contagens já calculada, inclusive DataFrame lido do CSV do projeto.
        labels: Ordem das classes. Em arrays de contagens, use a mesma ordem de construção.
        normalize: None para contagens; true por linha, pred por coluna, all pelo total.
        cmap, ax, figsize: Mapa de cores, eixo e tamanho da figura.
        **options: Opções de plot_heatmap/finish_plot, como annotate, rotation e save_path.

    Returns:
        Tupla (fig, ax). Divisões com denominador zero ficam em zero; entrada não é alterada.
    """
    if normalize not in {None, "true", "pred", "all"}:
        raise ValueError("normalize deve ser None, 'true', 'pred' ou 'all'")
    
    if matrix is not None:
        if y_true is not None or y_pred is not None:
            raise ValueError("Informe matrix OU y_true/y_pred")
        
        if isinstance(matrix, pd.DataFrame):
            if matrix.index.has_duplicates or matrix.columns.has_duplicates or list(matrix.index) != list(matrix.columns):
                raise ValueError("O DataFrame de contagens deve ter as mesmas classes/ordem nos dois eixos")
            
            if labels is not None and list(labels) != list(matrix.index):
                raise ValueError("labels difere da ordem das classes na matriz")
            
            labels = list(matrix.index)

        values = np.asarray(matrix, dtype=float).copy()
        if values.ndim != 2 or not values.size or values.shape[0] != values.shape[1]:
            raise ValueError("A matriz de confusão deve ser quadrada e não vazia")
        
        labels = list(range(len(values))) if labels is None else list(labels)
    else:
        true, predicted = np.asarray(y_true), np.asarray(y_pred)
        if true.ndim != 1 or predicted.shape != true.shape or not len(true):
            raise ValueError("y_true/y_pred devem ser vetores não vazios do mesmo tamanho")
        
        if pd.isna(true).any() or pd.isna(predicted).any():
            raise ValueError("Rótulos não podem conter nulos")
        
        observed = pd.unique(np.concatenate([true, predicted])).tolist()
        labels = observed if labels is None else list(labels)

        if not set(observed).issubset(labels):
            raise ValueError("labels não cobre todas as classes reais e previstas")
        
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("labels deve conter classes distintas")
        
        values = confusion_matrix(true, predicted, labels=labels).astype(float)
    if len(labels) != len(values) or len(set(labels)) != len(labels):
        raise ValueError("labels deve corresponder à matriz, sem repetições")
    
    if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError("Forneça contagens inteiras, finitas e não negativas; normalize nesta função")
    
    if normalize is not None:
        divisor = values.sum(axis=1, keepdims=True) if normalize == "true" else (
            values.sum(axis=0, keepdims=True) if normalize == "pred" else values.sum())
        values = np.divide(values, divisor, out=np.zeros_like(values), where=divisor != 0)

    return plot_heatmap(values, row_labels=labels, column_labels=labels, cmap=cmap, ax=ax,
                        figsize=figsize, **defaults(options, title="Matriz de confusão", xlabel="Previsto",
                        ylabel="Real", value_format=".2f" if normalize else ".0f",
                        vmin=0, vmax=1 if normalize else None))

def _score_curves(y_true, scores, classes: Sequence, positive_label=None):
    """Valida escores e devolve pares binários por classe, respeitando a ordem informada."""
    classes = list(classes)

    true, values = np.asarray(y_true), np.asarray(scores, dtype=float)
    if len(classes) < 2 or len(set(classes)) != len(classes) or pd.isna(classes).any():
        raise ValueError("classes deve ter pelo menos duas classes distintas, sem nulos")
    
    if true.ndim != 1 or not len(true) or pd.isna(true).any() or not set(true).issubset(classes):
        raise ValueError("y_true inválido ou contém classe não declarada")
    
    if values.ndim not in {1, 2} or len(values) != len(true) or not np.isfinite(values).all():
        raise ValueError("Escores devem ser finitos e ter uma linha por observação")
    
    if values.ndim == 1:
        if len(classes) != 2:
            raise ValueError("Escores 1D exigem exatamente duas classes")
        
        positive_label = classes[1] if positive_label is None else positive_label
        if positive_label not in classes:
            raise ValueError("positive_label não pertence às classes")
        
        curves = [(positive_label, values)]
    else:
        if values.shape[1] != len(classes) or positive_label is not None:
            raise ValueError("Escores 2D exigem uma coluna por classe e positive_label=None")
        
        curves = [(label, values[:, index]) for index, label in enumerate(classes)]

    result = []
    for label, score in curves:
        binary = (true == label).astype(int)
        if binary.min() == binary.max():
            raise ValueError(f"Classe {label!r} exige exemplos positivos e negativos para calcular a curva")
        
        result.append((label, binary, score))

    return result

def plot_roc_curve(y_true, scores, *, classes: Sequence, positive_label=None,
                   colors: Mapping | None = None, baseline: bool = True,
                   line_kwargs: dict | None = None, ax: Axes | None = None,
                   figsize: tuple = (7, 6), **options) -> PlotResult:
    """Desenha ROC binária ou uma curva one-vs-rest por classe com AUC na legenda.

    Args:
        y_true: Rótulos reais, na mesma ordem das linhas de scores.
        scores: Vetor de escores positivos (binário) ou matriz (N, C), sem arredondar.
        classes: Classes na ordem EXATA das colunas da matriz; obrigatório.
        positive_label: Somente para scores 1D; padrão classes[1].
        colors, baseline, line_kwargs: Mapa classe -> cor, diagonal de referência e opções de Axes.plot.
        ax, figsize, **options: Eixo, tamanho e opções comuns de finish_plot.

    Returns:
        Tupla (fig, ax). Rejeita classes sem positivos/negativos; não preenche AUC indefinida.
    """
    curves = _score_curves(y_true, scores, classes, positive_label)
    ax = create_axes(ax, figsize)

    for label, binary, score in curves:
        fpr, tpr, _ = roc_curve(binary, score)
        kwargs = {"label": f"{label} (AUC={auc(fpr, tpr):.3f})", **(line_kwargs or {})}

        if colors and label in colors:
            kwargs["color"] = colors[label]

        ax.plot(fpr, tpr, **kwargs)

    if baseline:
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Referência aleatória")

    return finish_plot(ax, **defaults(options, title="Curva ROC", xlabel="Taxa de falsos positivos",
                        ylabel="Taxa de verdadeiros positivos", xlim=(0, 1), ylim=(0, 1.03), legend=True))

def plot_precision_recall_curve(y_true, scores, *, classes: Sequence, positive_label=None,
                                colors: Mapping | None = None, baseline: bool = False,
                                line_kwargs: dict | None = None, ax: Axes | None = None,
                                figsize: tuple = (7, 6), **options) -> PlotResult:
    """Desenha Precision-Recall binária ou one-vs-rest, com Average Precision na legenda.

    Args:
        y_true, scores, classes, positive_label: Mesmo contrato explícito de plot_roc_curve.
        colors: Mapa classe -> cor, opcional.
        baseline: Desenha uma linha de prevalência para cada classe exibida.
        line_kwargs: Opções de Axes.plot, como linewidth, linestyle e alpha.
        ax, figsize, **options: Eixo, tamanho e opções comuns de finish_plot.

    Returns:
        Tupla (fig, ax). AP é average_precision_score, não área trapezoidal sob a curva.
    """
    curves = _score_curves(y_true, scores, classes, positive_label)
    ax = create_axes(ax, figsize)

    for label, binary, score in curves:
        precision, recall, _ = precision_recall_curve(binary, score)
        kwargs = {"label": f"{label} (AP={average_precision_score(binary, score):.3f})", **(line_kwargs or {})}

        if colors and label in colors:
            kwargs["color"] = colors[label]

        line, = ax.plot(recall, precision, **kwargs)

        if baseline:
            ax.axhline(binary.mean(), linestyle="--", color=line.get_color(), alpha=0.5,
                       label=f"Prevalência: {label}")
            
    return finish_plot(ax, **defaults(options, title="Curva Precision-Recall", xlabel="Recall",
                        ylabel="Precisão", xlim=(0, 1), ylim=(0, 1.03), legend=True))
