# Experimentos e Limitações

## Registro de evolução

Durante o desenvolvimento, a arquitetura de enriquecimento passou por revisões de sinais, hiperparâmetros e contratos de saída. Os resultados dessa etapa são preliminares e não substituem uma comparação final sobre o mesmo conjunto de teste.

### Visão computacional

Foi construída uma etapa de *image captioning* baseada em BLIP. A qualidade das legendas, sua granularidade e o custo computacional precisam ser confrontados com o ganho real no classificador final.

### PLN

A primeira abordagem usava sentimento, tópico principal, margem semântica e urgência. A classificação de tópico apresentou resultados instáveis, incluindo grande quantidade de registros com interpretação indefinida. Em resposta, foram discutidos e explorados sinais compactos de intenção textual, relacionados a falha, solicitação de acesso, dúvidas e lentidão.

A alternativa apresentou resultados promissores em experimentos preliminares de validação, mas continua sendo uma candidata, ainda é necessário decidir as colunas finais, revisar os contratos e realizar avaliação controlada. Se esse documento não for atualizado até a entrega, significa que não houve sucesso nessa tentativa.

### Deep learning tabular

O agrupamento por representação latente foi implementado para criar um perfil categórico. A interpretação dos clusters e seu ganho incremental precisam ser demonstrados por ablação.

### Multimodal

Foram experimentados classificadores tabulares de fusão, incluindo MLP e regressão logística. Uma diferença substancial entre validação e teste em experimento anterior destacou a necessidade de investigar generalização e o protocolo de seleção. Em termos informais, tá foda melhorar.

## Limitações conhecidas

- A representatividade do dataset frente a chamados reais ainda precisa ser estabelecida.
- Qualidade das imagens e dos títulos/descrições pode variar bastante.
- Scores semânticos dependem das hipóteses ou representações escolhidas.
- Clusters não são necessariamente categorias interpretáveis.
- Variáveis disponíveis somente após a resolução não podem ser usadas em triagem inicial.
- Muitas tentativas sobre a mesma validação podem gerar seleção excessiva de modelos.
- Melhoras em validação não comprovam melhoras em teste.

## Modelo de registro para novos experimentos

Para cada experimento, documentar: identificador de execução, versão do dataset, split/seed, features incluídas, parâmetros, métricas de validação, métricas de teste (apenas quando aplicável), artefatos e interpretação do resultado.

[PLN](../models/PLN.md) · [Avaliação](AVALIACAO.md) · [Índice](../README.md)
