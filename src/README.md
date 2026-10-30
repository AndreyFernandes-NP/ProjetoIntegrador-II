# Código-fonte (`src/`)

Reúne a implementação do processamento, da extração de características e da modelagem multimodal. A divisão em módulos busca manter responsabilidades claras e permitir a substituição de um gerador sem reescrever todo o fluxo.

## Organização

- [`config/`](config/README.md): parâmetros e definições declarativas.
- [`data/`](data/README.md): leitura, transformações, validação e divisão do dataset.
- [`models/`](models/README.md): visão computacional, PLN, deep learning tabular, enriquecimento e fusão multimodal.

Os nomes exatos dos scripts dependem da revisão do desenvolvimento local. Consulte também o [mapa completo de pastas](../docs/architecture/ESTRUTURA_DE_PASTAS.md).

## Critérios de manutenção

- Centralizar parâmetros estáveis nos YAMLs.
- Manter contratos explícitos de entrada e saída.
- Separar fit/treinamento de transform/inferência.
- Persistir artefatos suficientes para repetição das experiências.
- Impedir vazamento da variável-alvo entre etapas.

[Arquitetura](../docs/architecture/ARQUITETURA.md) · [Pipeline](../docs/architecture/PIPELINE.md) · [README principal](../README.md)
