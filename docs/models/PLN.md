# Processamento de Linguagem Natural (PLN)

## Objetivo

Aproveitar conjuntamente `title` e `description` para extrair características semânticas que ajudem o classificador multimodal. A etapa não precisa predizer diretamente a categoria `target_category`.

## Implementação de referência

O primeiro gerador trabalhou com um modelo de inferência textual multilíngue (`MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli`). A implementação foi organizada em `src/models/language.py` e configurada em `models.yaml`.

| Saída original | Objetivo | Situação |
| --- | --- | --- |
| `sentiment_score` | Estimar polaridade do texto | Mantida como candidata |
| `main_topic` | Inferir assunto principal | Sob revisão: excesso de rótulos indefinidos / inconsistentes |
| `semantic_margin` | Distância entre duas opções de tópico | Dependente da qualidade do tópico |
| `urgency_score` | Estimar urgência em contexto | Utilidade a reavaliar |

## Alternativa em estudo: sinais de intenção

Em vez de tentar classificar o assunto, a alternativa experimental busca medir indícios de:

- **Falha:** relato de problema ou impossibilidade de uso.
- **Solicitação de acesso:** pedido de autenticação, permissão ou liberação.
- **Dúvida ou orientação:** solicitação de explicação ou instrução.
- **Lentidão:** relato de baixo desempenho ou demora.

Os nomes definitivos das colunas e a combinação com `sentiment_score` ainda precisam ser fechados. O critério é gerar poucas variáveis úteis e evitar que a etapa vire uma segunda classificação da categoria final.

## Critérios de escolha

1. Cada sinal precisa ter interpretação clara e utilizar o texto de entrada.
2. As saídas devem manter tipos e intervalos definidos.
3. Não usar `target_category` ou conteúdo posterior à triagem como entrada.
4. Comparar a versão candidata à referência com os mesmos splits.
5. Verificar desempenho por categoria e contribuição marginal dos sinais.

[Contratos](CONTRATOS_DE_FEATURES.md) · [Experimentos](../analysis/EXPERIMENTOS_E_LIMITACOES.md) · [Índice](../README.md)
