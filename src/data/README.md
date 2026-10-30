# Processamento de Dados (`src/data/`)

Contém os componentes responsáveis por transformar a base de entrada em dados validados e particionados para modelagem.

## Módulos trabalhados no desenvolvimento

| Módulo | Função |
| --- | --- |
| `pipeline.py` | Coordenação do processamento |
| `processor.py` | Leitura de configuração, transformação e persistência |
| `validator.py` | Validação de conteúdo e qualidade |
| `splitter.py` | Treino, validação, teste e persistência de índices |

## Responsabilidades

- Preservar a integridade do dataset.
- Verificar contratos definidos em `sources.yaml`.
- Evitar ajustes de pré-processamento com informações do teste.
- Registrar o split usado pelos geradores e pela fusão.

[Tratamento](../../docs/data/TRATAMENTO_DE_DADOS.md) · [Divisão e validação](../../docs/data/DIVISAO_E_VALIDACAO.md) · [README principal](../../README.md)
