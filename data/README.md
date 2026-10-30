# Dados (`data/`)

Armazena as bases e os arquivos auxiliares usados na preparação e no enriquecimento dos chamados. O conteúdo deve ser organizado conforme a função dos dados, mantendo a entrada separada dos artefatos gerados.

## Organização de referência

- `clean/`: CSV tratado de entrada, como `chamados-clean.csv`.
- `processed/`: datasets padronizados em Parquet, como `chamados.parquet`.
- `processed/splits/`: índices persistidos da divisão entre treino, validação e teste.
- `annotations/`: anotações ou legendas de evidências visuais.
- Pastas de imagens: localizadas conforme os caminhos em `image_path` e a configuração do projeto.

## Diretrizes

- Não substituir a base de entrada por uma versão enriquecida.
- Conferir tipos, categorias e integridade com `sources.yaml` antes da modelagem.
- Registrar a versão da base e o split em cada experimento.

[Fontes de dados](../docs/data/FONTES_DE_DADOS.md) · [Dicionário](../docs/data/DICIONARIO_DE_DADOS.md) · [Tratamento](../docs/data/TRATAMENTO_DE_DADOS.md) · [README principal](../README.md)
