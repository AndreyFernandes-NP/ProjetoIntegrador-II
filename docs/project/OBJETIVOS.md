# Objetivos do Projeto

## Objetivo geral

Desenvolver e avaliar uma abordagem de classificação multimodal de chamados de TI, integrando características obtidas de dados tabulares, textos e evidências visuais para apoiar a triagem de solicitações.

## Objetivos específicos

1. Organizar o conjunto de dados, seus tipos, campos obrigatórios e possíveis inconsistências.
2. Definir divisão reproduzível em treino, validação e teste, evitando duplicação de evidências entre partições.
3. Extrair informações das imagens utilizando visão computacional e avaliar sua utilidade.
4. Gerar sinais a partir de título e descrição usando PLN, sem depender exclusivamente de uma classificação de assunto.
5. Aprender perfis tabulares por meio de uma abordagem de deep learning.
6. Integrar os sinais com os atributos originais em um classificador multimodal.
7. Comparar modelos e realizar ablações para medir a contribuição de cada modalidade.
8. Documentar decisões, incertezas e resultados de forma reproduzível.

## Resultados esperados

- Dataset processado, validado e enriquecido.
- Geradores de features rastreáveis e reaproveitáveis.
- Experimentos comparáveis, com métricas de classificação e análise por categoria.
- Evidência empírica sobre a contribuição — ou ausência de contribuição — de cada modalidade.
- Documentação e artefatos suficientes para o relatório acadêmico.

**Importante:** os resultados esperados são metas, não afirmações de desempenho já comprovado.

[Escopo](ESCOPO.md) · [Metodologia](../analysis/METODOLOGIA.md) · [Índice](../README.md)
