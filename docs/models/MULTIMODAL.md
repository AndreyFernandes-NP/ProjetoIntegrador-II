# Classificador Multimodal

## Propósito

Combinar informações provenientes de texto, evidências visuais e variáveis tabulares para predizer a categoria do chamado (`target_category`). A classificação final pertence a unicamente essa etapa.

## Estratégia

A arquitetura estudada segue *early fusion*: os atributos originais permitidos e as features geradas são reunidos em uma única representação tabular, usada por um classificador supervisionado.

## Busca da MLP

São amostradas 12 combinações com seed 42, selecionadas por F1 macro na validação. Cada tentativa ajusta o pré-processamento apenas no treino, utiliza AdamW, ReLU, dropout e clipping dos gradientes e restaura a melhor época antes da avaliação. O limite é de 40 épocas, com parada após 6 épocas sem melhora superior a `min_delta: 0.0001`.

| Parâmetro | Candidatos |
| --- | --- |
| Camadas ocultas | `[64]`, `[64, 32]`, `[128, 64]` |
| Dropout | `0.1`, `0.3`, `0.5` |
| Learning rate | `0.0003`, `0.001` |
| Weight decay | `0.0`, `0.0001`, `0.001` |
| Batch size | `32`, `64`, `128` |
| Pesos de classes | Sem ponderação ou balanceados |

A busca também considera as configurações de TF-IDF já existentes. As 12 tentativas são uma amostra do espaço de combinações, não uma avaliação de todas elas. Os intervalos são candidatos para experimentação; não representam uma melhoria de acurácia já demonstrada no dataset atual.

Uma nova busca pode reutilizar a execução de features concluída. Não é necessário gerar novamente visão, PLN, perfis ou split apenas para mudar o classificador. A quantidade total de tentativas foi mantida, mas executar 12 MLPs pode demorar mais que a busca anterior com 6 MLPs e 6 regressões logísticas.

## Entradas esperadas

- Variáveis originais que já existem no momento da triagem.
- Sinais visuais extraídos por visão computacional.
- Sinais textuais gerados a partir de título e descrição.
- Perfis produzidos pelo gerador tabular.

O alvo não entra no vetor de atributos. Identificadores arbitrários e dados posteriores à resolução do chamado também devem ser excluídos.

## Condições para uma comparação justa

1. Manter a mesma divisão entre configurações comparadas.
2. Ajustar pré-processadores somente com o conjunto de treino.
3. Comparar com uma baseline sem enriquecimento.
4. Realizar ablações para identificar o valor de cada modalidade.
5. Usar métricas macro e por categoria, além da acurácia.
6. Reservar teste para a avaliação final da configuração selecionada.

## Problema observado no desenvolvimento

Houve desempenho de validação significativamente superior ao observado em teste em uma versão anterior. Isso exige investigar distribuição, generalização e possíveis fontes de contaminação, sem assumir previamente uma única causa.

[Contratos](CONTRATOS_DE_FEATURES.md) · [Avaliação](../analysis/AVALIACAO.md) · [Índice](../README.md)
