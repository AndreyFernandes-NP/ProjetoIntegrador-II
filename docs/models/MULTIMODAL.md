# Classificador Multimodal

## Propósito

Combinar informações provenientes de texto, evidências visuais e variáveis tabulares para predizer a categoria do chamado (`target_category`). A classificação final pertence a unicamente essa etapa.

## Estratégia

A arquitetura estudada segue *early fusion*: os atributos originais permitidos e as features geradas são reunidos em uma única representação tabular, usada por um classificador supervisionado.

Foram trabalhadas alternativas como regressão logística e MLP como apresentações mais adequadas para classificação, ambas podem ser muito bem utilizadas, porém pra esse caso, seguimos com uma MLP para os resultados finais.

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
