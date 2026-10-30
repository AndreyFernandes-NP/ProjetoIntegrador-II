# Pipeline do Projeto

## Visão geral

O fluxo proposto transforma a base de chamados em uma representação multimodal adequada à classificação. A execução efetiva é configurada pelo código e pelos YAMLs, esse documento descreve apenas as responsabilidades e dependências, não comandos de terminal a serem executados.

## 1. Preparação da base

1. Ler a entrada definida em `sources.yaml`.
2. Padronizar tipos e remover colunas configuradas.
3. Validar obrigatoriedade, valores permitidos, intervalos e integridade.
4. Persistir o dataset processado em Parquet.

## 2. Particionamento

1. Identificar registros por `ticket_id`.
2. Construir/reutilizar divisões de treino, validação e teste.
3. Aplicar agrupamento de evidências por hash de imagem quando configurado.
4. Registrar seed e critérios da divisão.

## 3. Enriquecimento por modalidade

| Etapa | Entrada | Saída de referência |
| --- | --- | --- |
| Visão computacional | `image_path` | `image_description`, `image_score` |
| PLN | `title`, `description` | `sentiment_score` e sinais textuais em revisão |
| Deep learning | Campos numéricos/categóricos selecionados | `tabular_profile_cluster` |

Na implementação discutida, `features_pipeline.py` coordena os geradores, confere tipos e alinhamento dos índices e registra configurações e artefatos. O gerador tabular pode depender do sentimento atualizado pela etapa PLN.

## 4. Fusão multimodal

A base enriquecida reúne variáveis originais e sinais extraídos. Um classificador final é treinado com o alvo `target_category`, sem utilizar informações posteriores à triagem.

## 5. Avaliação e rastreabilidade

Os relatórios devem permitir recuperar o dataset, o split, a versão dos modelos, as configurações, as métricas e a seleção final. A interpretação dos resultados deve distinguir treino, validação e teste.

## 6. Fluxo de inferência futuro

Em uma versão estabilizada, novos chamados deverão passar pelas mesmas transformações e pelos modelos persistidos, sem o reajuste de componentes usando dados de inferência.

[Arquitetura](ARQUITETURA.md) · [Divisão](../data/DIVISAO_E_VALIDACAO.md) · [Multimodal](../models/MULTIMODAL.md) · [Índice](../README.md)
