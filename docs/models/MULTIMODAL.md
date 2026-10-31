# Classificador Multimodal

## Propósito

Combinar informações provenientes de texto, evidências visuais e variáveis tabulares para predizer a categoria do chamado (`target_category`). A classificação final pertence a unicamente essa etapa.

## Estratégia

A arquitetura estudada segue *early fusion*: os atributos originais permitidos e as features geradas são reunidos em uma única representação tabular, usada por um classificador supervisionado.

## Busca da MLP

Executa uma única amostra com seed 42, em caso de mais uma amostra, elas são selecionadas por F1 macro na validação. É possível configurar o `models.yaml` para experimentar com diversas amostras e parâmetros diferentes, cada tentativa ajusta o pré-processamento apenas no treino, o treinamento utiliza AdamW, ReLU, dropout, clipping dos gradientes e Regularização L2, restaurando a melhor época antes da avaliação. O limite é de 40 épocas, com parada após 6 épocas sem melhora superior a `min_delta: 0.0001`.

| Parâmetro | Candidatos |
| --- | --- |
| Camadas ocultas | `[64, 128]` |
| Dropout | `0.3` |
| Learning rate | `0.001` |
| Weight decay | `0.0001` |
| Batch size | `32` |
| Pesos de classes | Balanceados |

A busca também considera as configurações de TF-IDF já existentes.

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

Houve desempenho de validação significativamente superior ao observado em teste em uma versão anterior. A última técnica aplicada foi a Regularização L2, que por sua natureza de redução de pesos combinada com o Weight Decay, apesar de parecer uma interessante combinação, mostrou uma melhora significativa tanto para a generalização como f1-score do modelo, indicando que de fato os pesos estavam mais elevados.

[Contratos](CONTRATOS_DE_FEATURES.md) · [Avaliação](../analysis/AVALIACAO.md) · [Índice](../README.md)
