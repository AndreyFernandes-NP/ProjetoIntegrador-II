# Metodologia

## Abordagem geral

O projeto segue um desenvolvimento iterativo inspirado nas etapas do CRISP-DM: compreensão do problema, entendimento dos dados, preparação, modelagem, avaliação e revisão dos resultados.

## 1. Entendimento do problema

Caracterizar a triagem de chamados e definir o objetivo de previsão de `target_category` usando apenas informações disponíveis durante a solicitação.

## 2. Entendimento dos dados

Examinar distribuição de classes, valores ausentes, qualidade textual, evidências repetidas, campos temporais e atributos capazes de provocar vazamento.

## 3. Preparação

Aplicar conversões e validações declarativas, persistir a base em Parquet e definir partições reprodutíveis com agrupamento de evidências.

## 4. Extração de características

Explorar três modalidades de forma separada: descrição de imagens, características semânticas de texto e perfis tabulares por deep learning. Evitar o uso do alvo nos geradores.

## 5. Classificação

Combinar features selecionadas em modelos supervisionados e comparar com baselines. Manter controles de versão de dados, splits, artefatos e hiperparâmetros.

## 6. Avaliação e discussão

Utilizar métricas macro, por classe e matrizes de confusão; realizar ablação das modalidades e examinar diferenças de desempenho entre validação e teste.

## Caráter iterativo

Uma feature semanticamente plausível pode não ser preditivamente útil. A metodologia prevê retornar à seleção e à implementação dos sinais sempre que os resultados indicarem ruído ou inconsistência.

[Hipóteses](HIPOTESES.md) · [Avaliação](AVALIACAO.md) · [Índice](../README.md)
