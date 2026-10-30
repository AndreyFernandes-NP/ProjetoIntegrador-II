"""
distributions.py
----------------
Gráficos de distribuição numérica, frequência de categorias e dados ausentes,
com seleção de colunas e opções de apresentação reutilizáveis pro dataset.
"""
from collections.abc import Sequence

from matplotlib.axes import Axes
import numpy as np
import pandas as pd

from .base import PlotResult, create_axes, finish_plot, numeric_frame, require_columns, positive_top_n, defaults

def plot_histogram(data: pd.DataFrame, column: str, *, bins: int | str | Sequence = 30,
                   density: bool = False, color: str = "#377eb8", alpha: float = 0.85,
                   hist_kwargs: dict | None = None, ax: Axes | None = None,
                   figsize: tuple = (8, 5), **options) -> PlotResult:
    """Plota a distribuição de uma coluna numérica, removendo apenas seus valores nulos.

    Args:
        data, column: DataFrame de origem e coluna numérica; os dados não são modificados.
        bins: Número de intervalos, bordas explícitas ou regra do Matplotlib, como 'auto'.
        density: Exibe densidade (área total 1) em vez de contagens.
        color, alpha, hist_kwargs: Aparência e argumentos adicionais de Axes.hist.
        ax, figsize: Eixo existente ou tamanho de uma figura nova, em polegadas.
        **options: Opções de finish_plot, incluindo title, xlabel, save_path e show.

    Returns:
        Tupla (fig, ax). Dados vazios ou infinitos geram ValueError.
    """
    values = numeric_frame(data, [column])[column]
    ax = create_axes(ax, figsize)
    kwargs = {"bins": bins, "density": density, "color": color, "alpha": alpha}
    kwargs.update(hist_kwargs or {})
    ax.hist(values, **kwargs)

    return finish_plot(ax, **defaults(options, xlabel=column, ylabel="Densidade" if kwargs["density"] else "Frequência"))

def plot_boxplot(data: pd.DataFrame, columns: Sequence[str], *, horizontal: bool = False,
                 showfliers: bool = True, color: str = "#80b1d3", box_kwargs: dict | None = None,
                 ax: Axes | None = None, figsize: tuple = (8, 5), **options) -> PlotResult:
    """Compara distribuições numéricas com boxplots, removendo nulos por coluna.

    Args:
        data, columns: DataFrame e sequência não vazia de colunas numéricas.
        horizontal, showfliers: Orientação e exibição dos pontos além dos bigodes.
        color, box_kwargs: Cor das caixas e opções adicionais de Axes.boxplot.
        ax, figsize: Eixo existente ou tamanho da figura nova.
        **options: Personalização e salvamento via finish_plot.

    Returns:
        Tupla (fig, ax). Uma coluna inteiramente nula é rejeitada explicitamente.
    """
    if isinstance(columns, str):
        raise ValueError("columns deve ser uma sequência de nomes, não uma string")
    
    columns = list(columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("columns deve ser uma sequência não vazia de nomes distintos")
    
    arrays = [numeric_frame(data, [c])[c].to_numpy() for c in columns]
    ax = create_axes(ax, figsize)

    kwargs = {"orientation": "horizontal" if horizontal else "vertical", "showfliers": showfliers,
              "tick_labels": list(columns), "patch_artist": True, "boxprops": {"facecolor": color}}
    
    kwargs.update(box_kwargs or {})
    ax.boxplot(arrays, **kwargs)

    return finish_plot(ax, **options)

def plot_category_counts(data: pd.DataFrame, column: str, *, top_n: int | None = None,
                         normalize: bool = False, include_missing: bool = False,
                         missing_label: str = "(ausente)", horizontal: bool = True,
                         color: str = "#377eb8", ax: Axes | None = None,
                         figsize: tuple = (8, 5), **options) -> PlotResult:
    """Exibe frequências decrescentes de categorias, opcionalmente em porcentagem.

    Args:
        data, column: DataFrame e coluna categórica.
        top_n: Mostra apenas as N categorias mais frequentes; None mostra todas.
        normalize: Percentual do total considerado ANTES de limitar por top_n.
        include_missing, missing_label: Inclui nulos como uma categoria com rótulo próprio.
        horizontal, color: Orientação e cor das barras.
        ax, figsize, **options: Eixo, tamanho e opções de finish_plot.

    Returns:
        Tupla (fig, ax). Categorias omitidas pelo top_n não são redistribuídas.
    """
    require_columns(data, [column])
    positive_top_n(top_n)
    values = data[column]

    if include_missing:
        if values.dropna().eq(missing_label).any():
            raise ValueError("missing_label já existe nos dados; escolha outro rótulo")
        
        values = values.astype(object).where(values.notna(), missing_label)

    counts = values.value_counts(dropna=True)
    counts = counts[counts > 0]

    if counts.empty:
        raise ValueError("Não há categorias para exibir")
    
    if normalize:
        counts = counts / counts.sum() * 100

    counts = counts.head(top_n) if top_n else counts
    positions = np.arange(len(counts))
    ax = create_axes(ax, figsize)

    if horizontal:
        ax.barh(positions, counts.to_numpy(), color=color)
        ax.set_yticks(positions, labels=counts.index.astype(str))
        ax.set_ylim(len(counts) - 0.5, -0.5)
    else:
        ax.bar(positions, counts.to_numpy(), color=color)
        ax.set_xticks(positions, labels=counts.index.astype(str))

    measure = "Percentual (%)" if normalize else "Quantidade"
    return finish_plot(ax, **defaults(options, xlabel=measure if horizontal else column, ylabel=column if horizontal else measure))

def plot_missing_values(data: pd.DataFrame, *, columns: Sequence[str] | None = None,
                        top_n: int | None = None, percent: bool = True, only_missing: bool = True,
                        color: str = "#e69f00", ax: Axes | None = None,
                        figsize: tuple = (8, 5), **options) -> PlotResult:
    """Mostra valores ausentes por coluna, em ordem decrescente, sem alterar o dataset.

    Args:
        data, columns: DataFrame e colunas opcionais; None seleciona todas.
        top_n: Limita a quantidade de colunas exibidas.
        percent: Usa porcentagem de linhas ausentes em vez de contagem.
        only_missing: Omite colunas sem nulos. Se nenhuma restar, exibe uma mensagem.
        color, ax, figsize, **options: Aparência, eixo e opções de finish_plot.

    Returns:
        Tupla (fig, ax). DataFrame sem linhas é rejeitado, pois o percentual é indefinido.
    """
    columns = list(data.columns if columns is None else columns)

    require_columns(data, columns)
    positive_top_n(top_n)

    if data.empty or not columns:
        raise ValueError("Informe dados com linhas e colunas")
    
    values = data[columns].isna().sum().sort_values(ascending=False)
    if percent:
        values = values / len(data) * 100
    if only_missing:
        values = values[values > 0]

    values = values.head(top_n) if top_n else values
    ax = create_axes(ax, figsize)

    if values.empty:
        ax.text(0.5, 0.5, "Nenhum valor ausente", transform=ax.transAxes, ha="center", va="center")
    else:
        positions = np.arange(len(values))
        ax.barh(positions, values.to_numpy(), color=color)
        ax.set_yticks(positions, labels=values.index.astype(str))
        ax.set_ylim(len(values) - 0.5, -0.5)
        
    return finish_plot(ax, **defaults(options, xlabel="Ausentes (%)" if percent else "Quantidade de ausentes"))
