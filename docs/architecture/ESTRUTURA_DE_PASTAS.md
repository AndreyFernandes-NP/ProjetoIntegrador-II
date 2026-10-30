# Estrutura de Pastas

Esse documento descreve a finalidade de cada área do repositório de maneira geral. A árvore é uma referência da versão em desenvolvimento e inclui diretórios de saída que podem ser gerados sob demanda ou não estarem presentes.

## Diretórios principais

| Pasta | Conteúdo | Documentação local |
| --- | --- | --- |
| `data/` | Bases tratadas, processadas, imagens e anotações | [Abrir](../../data/README.md) |
| `docs/` | Planejamento, dados, arquitetura, modelos e análise | [Abrir](../README.md) |
| `notebooks/` | Exploração visual e testes independentes | [Abrir](../../notebooks/README.md) |
| `src/` | Implementação em Python | [Abrir](../../src/README.md) |
| `reports/` | Relatórios, métricas e saídas de experimentos | [Abrir](../../reports/README.md) |

## Subpastas de dados

- `data/clean/`: base de entrada tratada.
- `data/processed/`: datasets em Parquet e partições persistidas.
- `data/processed/splits/`: índices de partição.
- `data/annotations/`: legendas e anotações de imagens, quando utilizadas.
- Diretórios de imagens referenciados por `image_path`: localização depende da configuração.

## Subpastas do código

- `src/config/`: definições globais e YAMLs para dados e modelos.
- `src/data/`: `pipeline.py`, `processor.py`, `validator.py`, `splitter.py` e utilitários correlatos.
- `src/models/`: geradores `vision.py`, `language.py`, `profiles.py`, utilitários, enriquecimento e módulos de fusão.

## Artefatos que podem surgir em execução

- `models/huggingface/`: cópias locais dos pesos pré-treinados.
- `models/runs/<execucao>/`: pesos e metadados ajustados.
- `reports/runs/<execucao>/`: configurações, métricas, manifesto e dataset enriquecido.

[Arquitetura](ARQUITETURA.md) · [Pipeline](PIPELINE.md) · [Índice](../README.md)
