"""
training.py
-----------
Gráficos do histórico de treinamento, comparação de métricas e importância de features,
aceitando resultados já calculados sem carregar, ajustar ou executar modelos.
"""
from collections.abc import Mapping, Sequence

from matplotlib.axes import Axes
import numpy as np
import pandas as pd

from .base import PlotResult, create_axes, finish_plot, positive_top_n, defaults
from .relationships import plot_line, plot_bar

def plot_training_history(history, *, metric: str = "loss", validation: bool = True,
                          colors: Mapping[str, str] | None = None, start_epoch: int = 1,
                          ax: Axes | None = None, figsize: tuple = (8, 5), **options) -> PlotResult:
    """Compara a evolução de uma métrica de treino e sua validação, sem importar Keras.

    Args:
        history: Dicionário métrica -> sequência, ou objeto History com atributo history.
        metric: Métrica a exibir, como loss ou accuracy; deve existir no histórico.
        validation: Inclui val_<metric> quando disponível; não cria dados ausentes.
        colors: Mapa dos nomes metric/val_<metric> para cores.
        start_epoch: Número positivo da primeira época no eixo X.
        ax, figsize, **options: Eixo, tamanho e opções de plot_line/finish_plot.

    Returns:
        Tupla (fig, ax). Séries exibidas precisam ter o mesmo tamanho, não vazio.
    """
    values = getattr(history, "history", history)

    if not isinstance(values, Mapping) or metric not in values:
        raise ValueError(f"Histórico não contém a métrica {metric!r}")
    
    if type(start_epoch) is not int or start_epoch < 1:
        raise ValueError("start_epoch deve ser inteiro positivo")
    
    keys = [metric] + ([f"val_{metric}"] if validation and f"val_{metric}" in values else [])
    arrays = {key: np.asarray(values[key], dtype=float) for key in keys}

    if any(v.ndim != 1 or not len(v) or not np.isfinite(v).all() for v in arrays.values()) or len({len(v) for v in arrays.values()}) != 1:
        raise ValueError("Histórico deve ter séries finitas, não vazias e de mesmo tamanho")
    
    frame = pd.DataFrame(arrays)
    frame.insert(0, "epoch", np.arange(start_epoch, start_epoch + len(frame)))

    return plot_line(frame, "epoch", keys, colors=colors, ax=ax, figsize=figsize,
                     **defaults(options, title=f"Histórico de {metric}", xlabel="Época", ylabel=metric))

def plot_metric_comparison(results: Mapping[str, Mapping], *, metrics: Sequence[str] = ("accuracy", "f1_macro"),
                           colors: Mapping[str, str] | None = None, annotate: bool = True,
                           value_format: str = ".3f", ax: Axes | None = None,
                           figsize: tuple = (9, 5), **options) -> PlotResult:
    """Compara métricas entre modelos, execuções ou conjuntos em barras agrupadas.

    Args:
        results: Mapa nome -> métricas. Aceita o metrics.json atual (train/validation/test).
        metrics: Métricas escalares a comparar; todas devem existir em todos os grupos.
        colors: Mapa métrica -> cor; ausentes usam o ciclo do eixo.
        annotate, value_format: Mostra valores com o formato Python indicado.
        ax, figsize, **options: Eixo, tamanho e opções comuns de finish_plot.

    Returns:
        Tupla (fig, ax). Não limita automaticamente o eixo a 0..1: métricas podem ter outras unidades.
    """
    metrics = list(metrics)
    if not isinstance(results, Mapping) or not results or not metrics or len(set(metrics)) != len(metrics):
        raise ValueError("Informe resultados e métricas distintas, não vazios")
    
    names = list(results)
    matrix = []

    for name in names:
        if not isinstance(results[name], Mapping) or any(key not in results[name] for key in metrics):
            raise ValueError(f"Métricas ausentes em {name!r}")
        
        matrix.append([results[name][key] for key in metrics])

    values = np.asarray(matrix, dtype=float)
    if values.shape != (len(names), len(metrics)) or not np.isfinite(values).all():
        raise ValueError("Métricas devem ser valores escalares finitos")
    
    ax = create_axes(ax, figsize)
    positions = np.arange(len(names))
    width = 0.8 / len(metrics)

    for index, metric in enumerate(metrics):
        color = colors.get(metric) if colors else None
        bars = ax.bar(positions + (index - (len(metrics)-1)/2) * width, values[:, index],
                      width=width, label=metric, color=color)
        
        if annotate:
            ax.bar_label(bars, labels=[format(v, value_format) for v in values[:, index]], padding=3)

    ax.set_xticks(positions, labels=[str(name) for name in names])
    ax.margins(y=0.15)

    return finish_plot(ax, **defaults(options, ylabel="Valor da métrica", legend=True))

def plot_feature_importance(importances: Mapping[str, float] | pd.Series, *, top_n: int | None = 20,
                            sort_by_absolute: bool = True, color: str = "#009e73",
                            ax: Axes | None = None, figsize: tuple = (8, 6), **options) -> PlotResult:
    """Exibe importâncias já calculadas, mantendo sinais negativos e sem inferir causalidade.

    Args:
        importances: Dicionário ou Series com nomes de features e valores numéricos.
        top_n: Quantidade máxima de features; None mostra todas.
        sort_by_absolute: Ordena pela magnitude; False ordena pelo valor original.
        color, ax, figsize: Cor, eixo existente e tamanho da figura nova.
        **options: Opções de plot_bar/finish_plot, incluindo annotate e save_path.

    Returns:
        Tupla (fig, ax). Não calcula importância automaticamente para a rede multimodal.
    """
    positive_top_n(top_n)

    values = pd.Series(importances, dtype=float)
    if values.empty or values.index.has_duplicates or not np.isfinite(values.to_numpy()).all():
        raise ValueError("Importâncias devem ser finitas, não vazias e ter nomes únicos")
    
    order = (values.abs() if sort_by_absolute else values).sort_values(ascending=False).index
    ordered = values.loc[order]

    if top_n:
        ordered = ordered.head(top_n)

    frame = pd.DataFrame({"feature": ordered.index.astype(str), "importance": ordered.to_numpy()})
    frame = frame.iloc[::-1]
    
    return plot_bar(frame, "feature", "importance", horizontal=True, color=color, ax=ax, figsize=figsize,
                    **defaults(options, xlabel="Importância", ylabel="Feature"))
