# Contratos das Features

## Por que existem contratos

Cada gerador deve declarar suas entradas, saídas, tipos e dependências para que o pipeline identifique colunas inesperadas, alterações de índice, valores nulos e intervalos inválidos antes do treinamento final.

## Referência de saídas

| Etapa | Coluna | Tipo esperado | Ação |
| --- | --- | --- | --- |
| Visão | `image_description` | Texto não vazio | Adicionar |
| Visão | `image_score` | Numérico em `[0, 1]` | Adicionar |
| PLN | `sentiment_score` | Numérico em `[-1, 1]` | Substituir |
| PLN anterior | `main_topic` | Texto não vazio | Adicionar |
| PLN anterior | `semantic_margin` | Numérico em `[0, 1]` | Adicionar |
| PLN anterior | `urgency_score` | Numérico em `[0, 1]` | Adicionar |
| Deep learning | `tabular_profile_cluster` | Categoria nominal | Adicionar |
| PLN experimental | Sinais de intenção | Numéricos, intervalos a fixar | A definir |

## Controles essenciais

- **Alinhamento:** a saída mantém os mesmos índices do lote recebido.
- **Integridade:** sem valores ausentes ou não finitos onde o contrato exigir.
- **Tipo:** preservar texto, float e categórico conforme declarado.
- **Escopo:** não produzir ou consumir o alvo na etapa de enriquecimento.
- **Dependências:** executar etapas consumidoras após aquelas que fornecem suas colunas.
- **Reprodução:** persistir configurações e modelos usados na geração.

## Orçamento de dimensionalidade

A preferência do projeto é produzir aproximadamente quatro ou cinco novas features, evitando grande expansão do dataset. Uma descrição textual, quando utilizada como coluna, também deve ser contabilizada na avaliação da complexidade e do pré-processamento do multimodal.

## Alterações sincronizadas

Qualquer mudança de output deve ser refletida em `models.yaml`, nos contratos do pipeline, no gerador correspondente e no classificador que consome a feature. A revisão não pode deixar relatórios antigos indistinguíveis dos novos.

[PLN](PLN.md) · [Configurações](../architecture/CONFIGURACOES.md) · [Índice](../README.md)
