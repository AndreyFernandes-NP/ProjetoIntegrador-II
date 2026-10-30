# Visão Computacional

## Objetivo

Extrair informações visuais das evidências vinculadas aos chamados, fornecendo sinais complementares aos textos e às colunas tabulares.

## Abordagem em desenvolvimento

A configuração de referência utiliza o modelo pré-treinado `Salesforce/blip-image-captioning-base`, com adaptação para geração de legendas (*image captioning*).

| Item | Referência |
| --- | --- |
| Entrada | `image_path` |
| Saída textual | `image_description` |
| Saída numérica | `image_score` |
| Anotações opcionais | `data/annotations/image_captions.csv` |
| Implementação trabalhada | `src/models/vision.py` |

`image_description` resume elementos identificados na imagem, `image_score` representa um indicador relacionado à probabilidade da geração da legenda, tudo não deve ser interpretado como probabilidade de classificação correta do chamado.

## Fluxo lógico

1. Localizar e abrir a evidência visual.
2. Aplicar o processamento de imagem e texto exigido pelo modelo.
3. Gerar uma descrição resumida.
4. Produzir score e validar o formato das saídas.
5. Adicionar as colunas ao dataset enriquecido.

## Parâmetros de referência

A configuração discutida inclui duas épocas de ajuste supervisionado das legendas, batch de duas imagens, taxa de aprendizagem de `1e-5`, `num_beams: 3` e `max_new_tokens: 24`.

[Deep Learning](DEEP_LEARNING.md) · [Multimodal](MULTIMODAL.md) · [Índice](../README.md)
