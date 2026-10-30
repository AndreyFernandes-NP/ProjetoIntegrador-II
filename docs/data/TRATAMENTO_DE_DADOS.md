# Tratamento de Dados

## Visão geral

A preparação separa o CSV tratado (`data/clean/`) do dataset pronto para modelagem (`data/processed/`). O comportamento é definido em `src/config/sources.yaml` e implementado pelos componentes de `src/data/`.

## Transformações configuráveis

Na configuração de referência, as transformações incluem:

- Conversão de `ticket_id`, `customer_id`, `title`, `description`, `image_path` e `target_category` para texto.
- Conversão de `sla_hours`, `resolution_time_hours` e `sentiment_score` para números.
- Remoção explícita de colunas fora do recorte por meio de `drop_cols`.
- Persistência do dataset processado em Parquet.

A lista de campos descartados está centralizada no YAML.

## Validação da qualidade

As regras declarativas incluem:

1. Existência de colunas obrigatórias.
2. Ausência de nulos em campos essenciais.
3. Unicidade de `ticket_id`.
4. Verificação de campos numéricos e seus intervalos (ex.: `sentiment_score` entre -1 e 1).
5. Verificação de categorias permitidas em `target_category`.
6. Inspeção de proporção de nulos e integridade das imagens.

## Preservação e rastreabilidade

A base tratada não deve ser sobrescrita por resultados experimentais. O enriquecimento deve registrar configurações e artefatos por execução, permitindo comparar versões sem confundir a fonte com as saídas dos modelos.

[Fontes](FONTES_DE_DADOS.md) · [Divisão e validação](DIVISAO_E_VALIDACAO.md) · [Pipeline](../architecture/PIPELINE.md) · [Índice](../README.md)
