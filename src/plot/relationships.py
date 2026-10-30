"""
relationships.py
----------------
Gráficos de barras, dispersão, linhas, mapas de calor e correlação entre variáveis,
com suporte a eixos existentes e personalização.
"""
from collections.abc import Sequence, Mapping
from typing import Any, Literal

from matplotlib.axes import Axes
import numpy as np
import pandas as pd

from .base import PlotResult, create_axes, finish_plot, numeric_frame, require_columns, defaults

def plot_bar(data: pd.DataFrame, x: str, y: str, *, horizontal: bool = False,
             color: str = "#377eb8", annotate: bool = False, value_format: str = ".2f",
             bar_kwargs: dict | None = None, ax: Axes | None = None,
             figsize: tuple = (8, 5), **options) -> PlotResult:
    """Plota uma barra por linha de uma tabela já agregada, sem calcular médias ou somas.

    Args:
        data, x, y: Tabela, coluna de rótulos e coluna numérica de alturas.
        horizontal, color: Orientação e cor das barras.
        annotate, value_format: Mostra valores sobre barras usando formato Python, como '.1f'.
        bar_kwargs: Argumentos adicionais de Axes.bar/barh, como alpha e edgecolor.
        ax, figsize, **options: Eixo, tamanho e opções comuns de finish_plot.

    Returns:
        Tupla (fig, ax). Remove linhas com x/y nulos; rótulos repetidos permanecem separados.
    """
    if x == y:
        raise ValueError("Use colunas distintas para os rótulos e alturas das barras")
    
    require_columns(data, [x, y])
    frame = data[[x, y]].dropna().copy()
    values = numeric_frame(frame, [y])[y].to_numpy()
    labels = frame[x].astype(str).to_list()
    positions = np.arange(len(values))
    ax = create_axes(ax, figsize)

    kwargs = {"color": color, **(bar_kwargs or {})}

    if horizontal:
        bars = ax.barh(positions, values, **kwargs)
        ax.set_yticks(positions, labels=labels)
    else:
        bars = ax.bar(positions, values, **kwargs)
        ax.set_xticks(positions, labels=labels)

    if annotate:
        ax.bar_label(bars, labels=[format(v, value_format) for v in values], padding=3)
        ax.margins(x=0.15 if horizontal else 0.05, y=0.05 if horizontal else 0.15)

    return finish_plot(ax, **defaults(options, xlabel=y if horizontal else x, ylabel=x if horizontal else y))

def plot_scatter(data: pd.DataFrame, x: str, y: str, *, hue: str | None = None,
                 palette: Mapping | None = None, color: str = "#377eb8", size: float = 30,
                 alpha: float = 0.7, scatter_kwargs: dict[str, Any] | None = None,
                 ax: Axes | None = None, figsize: tuple = (8, 5), **options) -> PlotResult:
    """Mostra a relação de duas variáveis numéricas, com grupos categóricos opcionais.

    Args:
        data, x, y: DataFrame e nomes das variáveis numéricas.
        hue: Coluna de grupos; None usa uma cor única. Linhas com hue nulo são omitidas.
        palette: Mapa categoria -> cor. Categorias sem entrada usam o ciclo do eixo.
        color, size, alpha: Cor única, área dos marcadores em pontos² e transparência.
        scatter_kwargs: Argumentos adicionais de Axes.scatter, como marker e edgecolors.
        ax, figsize, **options: Eixo, tamanho e opções de finish_plot.

    Returns:
        Tupla (fig, ax). Nulos são removidos em conjunto para preservar os pares X/Y.
    """
    columns = list(dict.fromkeys([x, y] + ([hue] if hue else [])))

    require_columns(data, columns)
    frame = data[columns].dropna().copy()
    numeric = numeric_frame(frame, list(dict.fromkeys([x, y])))
    frame[x], frame[y] = numeric[x], numeric[y]
    ax = create_axes(ax, figsize)
    
    groups = frame.groupby(hue, sort=False, observed=True) if hue else [(None, frame)]
    for label, group in groups:
        kwargs: dict[str, Any] = {"s": size, "alpha": alpha}

        if hue is None:
            kwargs["color"] = color

        elif palette is not None and label in palette:
            kwargs["color"] = palette[label]

        if hue is not None:
            kwargs["label"] = str(label)

        kwargs.update(scatter_kwargs or {})
        ax.scatter(group[x], group[y], **kwargs)

    return finish_plot(ax, **defaults(options, xlabel=x, ylabel=y, legend=hue is not None))

def plot_line(data: pd.DataFrame, x: str, y: str | Sequence[str], *, sort_x: bool = True,
              colors: Mapping[str, str] | None = None, marker: str | None = None,
              line_kwargs: dict | None = None, ax: Axes | None = None,
              figsize: tuple = (8, 5), **options) -> PlotResult:
    """Desenha uma ou mais séries numéricas, inclusive com datas já convertidas no eixo X.

    Args:
        data, x, y: DataFrame, eixo X e uma coluna Y ou sequência de colunas Y.
        sort_x: Ordena X de forma estável; não agrega valores de X repetidos.
        colors, marker, line_kwargs: Mapa coluna -> cor, marcador e opções de Axes.plot.
        ax, figsize, **options: Eixo, tamanho e opções comuns de finish_plot.

    Returns:
        Tupla (fig, ax). Y nulo interrompe a linha; X nulo é removido.
    """
    columns = [y] if isinstance(y, str) else list(y)

    if not columns or len(set(columns)) != len(columns) or x in columns:
        raise ValueError("Informe colunas Y distintas entre si e diferentes de X")
    
    require_columns(data, [x] + columns)
    frame = data[[x] + columns].dropna(subset=[x]).copy()
    numeric = numeric_frame(frame, columns, dropna=False)

    if numeric.isna().all(axis=None):
        raise ValueError("Não há valores Y para exibir")
    
    frame[columns] = numeric
    if sort_x:
        frame = frame.sort_values(x, kind="stable")

    ax = create_axes(ax, figsize)
    for column in columns:
        kwargs = {"label": column, "marker": marker, **(line_kwargs or {})}

        if colors and column in colors:
            kwargs["color"] = colors[column]

        ax.plot(frame[x], frame[column], **kwargs)

    return finish_plot(ax, **defaults(options, xlabel=x, legend=True))

def plot_heatmap(matrix, *, row_labels: Sequence | None = None, column_labels: Sequence | None = None,
                 cmap: str = "viridis", annotate: bool = True, value_format: str = ".2f",
                 annotation_fontsize: float = 9, vmin: float | None = None, vmax: float | None = None,
                 colorbar: bool = True, colorbar_label: str = "", ax: Axes | None = None,
                 figsize: tuple = (8, 6), **options) -> PlotResult:
    """Plota uma matriz numérica; valores NaN aparecem mascarados, sem número na célula.

    Args:
        matrix: Array 2D ou DataFrame, cujos índices/colunas viram rótulos por padrão.
        row_labels, column_labels: Rótulos explícitos na mesma ordem da matriz.
        cmap, vmin, vmax: Mapa de cores e limites da escala.
        annotate, value_format, annotation_fontsize: Conteúdo e tamanho das anotações.
        colorbar, colorbar_label: Exibição e nome da escala lateral.
        ax, figsize, **options: Eixo, tamanho e opções comuns de finish_plot.

    Returns:
        Tupla (fig, ax). Infinito, matriz vazia e dimensões incompatíveis são rejeitados.
    """
    if isinstance(matrix, pd.DataFrame):
        row_labels = list(matrix.index) if row_labels is None else row_labels
        column_labels = list(matrix.columns) if column_labels is None else column_labels

        values = matrix.to_numpy(dtype=float, na_value=np.nan)
    else:
        values = np.asarray(matrix, dtype=float)

    if values.ndim != 2 or not values.size or np.isinf(values).any():
        raise ValueError("Informe uma matriz 2D não vazia, sem infinito")
    
    rows = list(range(values.shape[0])) if row_labels is None else list(row_labels)
    columns = list(range(values.shape[1])) if column_labels is None else list(column_labels)

    if len(rows) != values.shape[0] or len(columns) != values.shape[1]:
        raise ValueError("Quantidade de rótulos difere da matriz")
    
    ax = create_axes(ax, figsize)
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(columns)), labels=[str(c) for c in columns])
    ax.set_yticks(np.arange(len(rows)), labels=[str(r) for r in rows])

    if annotate:
        for (i, j), value in np.ndenumerate(values):
            if np.isfinite(value):
                r, g, b, _ = image.cmap(image.norm(value))
                text_color = "black" if 0.2126*r + 0.7152*g + 0.0722*b > 0.5 else "white"
                ax.text(j, i, format(value, value_format), ha="center", va="center",
                        color=text_color, fontsize=annotation_fontsize)
                
    if colorbar:
        figure = ax.get_figure(root=True)

        if figure is None:
            raise ValueError("O eixo precisa pertencer a uma figura")
        
        figure.colorbar(image, ax=ax, label=colorbar_label)

    return finish_plot(ax, **defaults(options, grid=False))

def plot_correlation(data: pd.DataFrame, columns: Sequence[str] | None = None, *,
                     method: Literal["pearson", "spearman", "kendall"] = "pearson",
                     min_periods: int = 2, cmap: str = "RdBu_r",
                     ax: Axes | None = None, figsize: tuple = (8, 6), **options) -> PlotResult:
    """Calcula correlação e a exibe como heatmap, preservando a ordem das colunas.

    Args:
        data, columns: DataFrame e colunas; None seleciona automaticamente apenas numéricas.
        method: pearson, spearman ou kendall.
        min_periods: Mínimo de observações válidas por par, pelo menos 2.
        cmap, ax, figsize: Mapa de cores, eixo e tamanho da figura.
        **options: Opções de plot_heatmap/finish_plot, como annotate, rotation e save_path.

    Returns:
        Tupla (fig, ax). Pares insuficientes e colunas constantes ficam com células NaN.
    """
    if method not in {"pearson", "spearman", "kendall"} or type(min_periods) is not int or min_periods < 2:
        raise ValueError("method inválido ou min_periods menor que 2")
    
    columns = list(data.select_dtypes(include="number").columns if columns is None else columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Selecione pelo menos uma coluna numérica, sem repetições")
    
    frame = numeric_frame(data, columns, dropna=False)
    matrix = frame.corr(method=method, min_periods=min_periods)
    
    return plot_heatmap(matrix, ax=ax, figsize=figsize, cmap=cmap,
                        **defaults(options, vmin=-1, vmax=1, colorbar_label=f"Correlação ({method})"))
