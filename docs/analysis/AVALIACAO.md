# Avaliação dos Modelos

## Objetivo

Medir qualidade de classificação, estabilidade e capacidade de generalização, distinguindo resultados de ajuste, seleção e avaliação final.

## Métricas prioritárias

| Métrica | Para que serve |
| --- | --- |
| Acurácia | Proporção global de categorias corretas |
| F1 macro | Média do F1 por classe com peso igual para cada classe |
| Precision/Recall por classe | Identificar erros específicos por categoria |
| Matriz de confusão | Visualizar trocas entre categorias |
| Distribuição por split | Detectar variações de classes e tamanho das partições |
| Tempo de execução | Medir custo computacional das alternativas |

O **F1 macro** é particularmente útil quando as categorias possuem quantidades diferentes de exemplos.

## Protocolo de comparação

1. Fixar a base e o split.
2. Definir uma baseline tabular/textual sem as novas features experimentais.
3. Executar variantes com visão, PLN e deep learning isoladamente.
4. Comparar combinações das modalidades.
5. Repetir experimentos quando necessário para avaliar estabilidade.
6. Selecionar parâmetros usando apenas treino/validação.
7. Avaliar uma configuração final no teste reservado.

## Análises adicionais

- Separar métricas por classe, em especial categorias com mais confusões.
- Investigar previsões com confiança alta e erro.
- Identificar classes raras, grupos de imagem repetidos e textos quase duplicados.
- Comparar os sinais novos às colunas já disponíveis para medir redundância.
- Relatar falhas observadas, não apenas a melhor pontuação.

## Interpretação dos resultados parciais

Há uma diferença expressiva na validação e no teste gerados, por esse motivo tentamos arduamente aprimorar e otimizar os modelos utilizados mas sem sucesso. Até então, esses valores pertencem a gerações antigas de split e podem ser diferentes com base nas configurações atuais após experimentações.

[Experimentos](EXPERIMENTOS_E_LIMITACOES.md) · [Divisão](../data/DIVISAO_E_VALIDACAO.md) · [Índice](../README.md)
