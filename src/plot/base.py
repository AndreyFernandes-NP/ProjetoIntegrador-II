"""
base.py
-------
Funções compartilhadas para criar eixos, validar dados e personalizar ou salvar figuras.
"""

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

PlotResult = tuple[Figure, Axes]

def create_axes(ax: Axes | None = None, figsize: tuple[float, float] = (8, 5)) -> Axes:
    """Reutiliza ax ou cria uma figura com figsize em polegadas; não limpa eixos existentes."""
    if ax is not None:
        return ax
    
    _, result = plt.subplots(figsize=figsize, layout="constrained")

    return result

def finish_plot(
    ax: Axes, *, title: str | None = None, xlabel: str | None = None,
    ylabel: str | None = None, xlim: tuple | None = None, ylim: tuple | None = None,
    xscale: str | None = None, yscale: str | None = None, grid: bool | None = None,
    grid_axis: Literal["both", "x", "y"] = "both", legend: bool | None = None, legend_kwargs: dict | None = None,
    rotation: float = 0, title_fontsize: float = 13, label_fontsize: float = 11,
    tick_fontsize: float = 10, save_path: str | Path | None = None, dpi: int = 150,
    transparent: bool = False, save_kwargs: dict | None = None, show: bool = False,
) -> PlotResult:
    """Aplica opções comuns e retorna (fig, ax), sem fechar a figura.

    Args:
        ax: Eixo que será configurado. A figura inteira é salva quando há subplots.
        title, xlabel, ylabel: Textos; None preserva o texto existente.
        xlim, ylim: Limites dos eixos; None mantém os limites atuais.
        xscale, yscale: Escalas do Matplotlib, como linear/log; None não altera.
        grid, grid_axis: Liga/desliga a grade e escolhe both/x/y; None preserva.
        legend, legend_kwargs: Controla a legenda; None preserva, True usa os labels.
        rotation: Rotação dos rótulos do eixo X, em graus.
        title_fontsize, label_fontsize, tick_fontsize: Tamanhos das fontes em pontos.
        save_path: Destino opcional. Caminhos relativos usam o diretório de execução.
        dpi, transparent, save_kwargs: Opções de Figure.savefig; save_kwargs tem prioridade.
        show: Exibe a figura via pyplot.show; False permite uso em scripts e notebooks.

    Returns:
        Tupla (Figure, Axes) para continuar editando. Diretórios são criados ao salvar.
    """
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_fontsize)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)

    if xlim is not None:
        ax.set_xlim(xlim)
        
    if ylim is not None:
        ax.set_ylim(ylim)

    if xscale is not None:
        ax.set_xscale(xscale)

    if yscale is not None:
        ax.set_yscale(yscale)
        
    if grid is not None:
        if grid:
            ax.grid(True, axis=grid_axis, alpha=0.25)
            ax.set_axisbelow(True)

        else:
            ax.grid(False, axis=grid_axis)

    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.tick_params(axis="x", labelrotation=rotation)

    if legend is True:
        handles, labels = ax.get_legend_handles_labels()

        if handles:
            ax.legend(**(legend_kwargs or {}))

    elif legend is False:
        current_legend = ax.get_legend()

        if current_legend is not None:
            current_legend.remove()

    fig = ax.get_figure(root=True)
    if not isinstance(fig, Figure):
        raise ValueError("O eixo precisa pertencer a uma figura")
    
    if save_path is not None:
        path = Path(save_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        settings = {"dpi": dpi, "transparent": transparent, "bbox_inches": "tight"}
        settings.update(save_kwargs or {})
        fig.savefig(path, **settings)

    if show:
        plt.show()

    return fig, ax

def require_columns(data: pd.DataFrame, columns: list[str]) -> None:
    """Rejeita colunas ausentes ou nomes duplicados para evitar seleção ambígua."""
    if not isinstance(data, pd.DataFrame) or data.columns.has_duplicates:
        raise ValueError("Informe um DataFrame com nomes de colunas únicos")
    
    absent = [column for column in columns if column not in data]
    if absent:
        raise ValueError(f"Colunas ausentes: {absent}")

def numeric_frame(data: pd.DataFrame, columns: list[str], *, dropna: bool = True) -> pd.DataFrame:
    """Copia/converte colunas numéricas; rejeita infinito e remove nulos pareados se solicitado."""
    require_columns(data, columns)
    result = data.loc[:, columns].apply(pd.to_numeric, errors="raise").astype(float)

    if np.isinf(result.to_numpy()).any():
        raise ValueError("Dados contêm infinito")
    
    if dropna:
        result = result.dropna()

    if result.empty:
        raise ValueError("Não há observações válidas para o gráfico")
    
    return result

def positive_top_n(top_n: int | None) -> None:
    """Valida um limite opcional de itens, sem aceitar zero, negativos ou booleanos."""
    if top_n is not None and (type(top_n) is not int or top_n < 1):
        raise ValueError("top_n deve ser um inteiro positivo ou None")

def defaults(options: dict[str, Any], **values: Any) -> dict[str, Any]:
    """Combina opções comuns sem sobrescrever escolhas explícitas do chamador."""
    
    return {**values, **options}
