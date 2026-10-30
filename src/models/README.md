# Modelos (`src/models/`)

Área de implementação das etapas de extração de características e da classificação multimodal. A arquitetura foi pensada para manter os geradores separados do classificador final.

## Módulos trabalhados / previstos na versão em desenvolvimento

| Módulo | Responsabilidade |
| --- | --- |
| `vision.py` | Gerar descrições e scores visuais |
| `language.py` | Extrair sinais de PLN a partir de título e descrição |
| `profiles.py` | Produzir perfis tabulares por deep learning |
| `feature_common.py` | Utilitários compartilhados |
| `features_pipeline.py` | Validar, organizar e persistir features enriquecidas |
| `reporting.py` | Metadados, relatórios e integridade de artefatos |
| `fusion.py` | Implementação do classificador multimodal |
| `fusion_pipeline.py` | Treino e avaliação da fusão |
| `fusion_predict.py` | Predição a partir de componentes salvos |
| `generator_tuning.py` | Experimentação e ajuste de geradores |

## Regras de integração

- Geradores não devem consumir `target_category`.
- Outputs precisam satisfazer os contratos de nomes, tipos e índices.
- Ajuste, transformação e restauração devem usar configurações consistentes.
- Novas colunas devem ser justificadas por utilidade preditiva e não por quantidade.

[Visão](../../docs/models/VISAO_COMPUTACIONAL.md) · [PLN](../../docs/models/PLN.md) · [Deep learning](../../docs/models/DEEP_LEARNING.md) · [Multimodal](../../docs/models/MULTIMODAL.md) · [README principal](../../README.md)
