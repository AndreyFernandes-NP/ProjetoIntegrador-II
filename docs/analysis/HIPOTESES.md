# Hipóteses do Projeto

As hipóteses abaixo são testáveis e não representam conclusões.

## Hipótese principal

**H1 — A integração de sinais textuais, visuais e tabulares melhora a classificação de chamados em relação a uma baseline construída apenas com os atributos originais disponíveis na triagem.**

**H0 — A integração não produz ganho consistente sob as mesmas condições de treino e avaliação.**

## Hipóteses secundárias

- **H2 (PLN):** sinais semânticos derivados de título e descrição contribuem mais que scores textuais pouco estáveis ou rótulos de tópico inconsistentes.
- **H3 (Visão):** evidências visuais relevantes adicionam informação não expressa integralmente no relato textual.
- **H4 (Tabular):** perfis aprendidos por deep learning capturam estruturas que auxiliam o classificador final.
- **H5 (Generalização):** ganhos aparentes em validação podem diminuir em teste quando existem diferenças de distribuição, duplicatas ou seleção excessiva de parâmetros.

## Como testar

Utilizar o mesmo particionamento e protocolo para todas as variantes. Comparar baseline, cada modalidade isolada e combinações; registrar F1 macro, desempenho por classe, estabilidade entre sementes e resultados finais em teste.

Não tratar uma diferença pontual em validação como confirmação de H1 ou de qualquer hipótese secundária.

[Metodologia](METODOLOGIA.md) · [Avaliação](AVALIACAO.md) · [Índice](../README.md)
