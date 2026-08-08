from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root:           Path   # ProjetoIntegrador-II/
    data:           Path   # data/
    data_raw:       Path   # data/raw/
    data_clean:     Path   # data/clean/
    data_processed: Path   # data/processed/
    data_images:    Path   # data/images/
#   reports:        Path   # reports/
    src:            Path   # src/
    core:           Path   # src/core/
    config:         Path   # src/config/
    src_data:       Path   # src/data/
    src_pipeline:   Path   # src/data/pipeline/
    notebooks:      Path   # notebooks/
    docs:           Path   # docs/


def get_project_paths(root: Path | None = None) -> ProjectPaths:
    project_root = root or Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    src_dir  = project_root / "src"

    return ProjectPaths(
        root           = project_root,
        data           = data_dir,
        data_raw       = data_dir / "raw",
        data_clean     = data_dir / "clean",
        data_processed = data_dir / "processed",
        data_images    = data_dir / "images",
#       reports        = project_root / "reports",
        src            = src_dir,
        core           = src_dir / "core",
        config         = src_dir / "config",
        src_data       = src_dir / "data",
        src_pipeline   = src_dir / "data" / "pipeline",
        notebooks      = project_root / "notebooks",
        docs           = project_root / "docs",
    )

# Instância global, só importar em qualquer módulo do projeto
PATHS = get_project_paths()