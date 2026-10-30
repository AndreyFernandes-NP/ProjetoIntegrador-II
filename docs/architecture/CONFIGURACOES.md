# Configurações do Projeto

## Princípio

As configurações separadas do código facilitam a revisão dos experimentos e diminuem a necessidade de alterar implementações para testar novas features ou modelos.

## `src/config/sources.yaml`

Define a origem e a saída do dataset, conversões de tipos, colunas descartadas e regras de qualidade. Na referência trabalhada:

- Entrada: `data/clean/chamados-clean.csv`.
- Saída: `data/processed/chamados.parquet`.
- Qualidade: obrigatoriedade, não nulos, unicidade, intervalos e categorias permitidas.

## `src/config/models.yaml`

Descreve os modelos utilizados no enriquecimento, seus campos de entrada, fontes de pesos, parâmetros, saídas e dependências entre etapas.

A configuração de referência utiliza `schema_version: 3` e três etapas, sob os nomes `visao_computacional`, `pln` e `deep_learning`. Os adaptadores de referência são `blip_captioning`, `contextual_nli` e `tabular_profiles`.

## Fonte de modelos

Os blocos `fonte.link` e `fonte.nome` identificam repositórios ou nomes de modelos pré-treinados. O armazenamento local de artefatos evita depender de mudanças posteriores dos pesos externos. Sempre registrar a revisão ou identificador do snapshot.

## Contratos e validações

Cada saída deve declarar nome, tipo, ação (`add` ou `replace`) e, quando aplicável intervalo. O pipeline valida a correspondência entre configuração, índice e valores entregues pelo gerador.

[Pipeline](PIPELINE.md) · [Contratos de features](../models/CONTRATOS_DE_FEATURES.md) · [Índice](../README.md)
