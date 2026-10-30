# Decisões e Diretrizes do Projeto

As decisões abaixo refletem a arquitetura e os critérios discutidos durante o desenvolvimento.

| Decisão | Justificativa | Situação |
| --- | --- | --- |
| YAML para processamento e modelos | Separar parâmetros e implementação | Adotada |
| Divisão reproduzível dos dados | Comparação justa entre experimentos | Adotada |
| Agrupamento por identidade/hash de imagem | Reduzir contaminação entre partições | Adotada |
| Geradores independentes da classificação final | Separar extração de sinais e decisão | Adotada |
| Poucas colunas de enriquecimento | Evitar expansão excessiva de dimensionalidade | Adotada |
| Revisão do `main_topic` | Classificações instáveis e muitos indefinidos | Em revisão |
| Sinais de intenção textual | Aproveitar título e descrição sem predizer categoria | Alternativa experimental |
| Documentar execução apenas ao final | Evitar instruções obsoletas | Adotada |

[Escopo](ESCOPO.md) · [Contratos de features](../models/CONTRATOS_DE_FEATURES.md) · [Índice](../README.md)
