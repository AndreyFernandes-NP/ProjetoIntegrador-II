# Divisão e Validação dos Dados

## Objetivo

Definir uma partição reproduzível que permita treinar e comparar modelos sem utilizar amostras de teste na seleção de features ou hiperparâmetros.

## Configuração de referência

| Parâmetro | Valor |
| --- | --- |
| Identificador | `ticket_id` |
| Semente | `42` |
| Treino | 70% |
| Validação | 15% |
| Teste | 15% |
| Estratégia | Agrupamento por evidência visual |
| Grupo | `image_path` com modo `image_hash` |
| Índices persistidos | `data/processed/splits/chamados.json` |

O agrupamento por hash busca impedir que imagens iguais ou equivalentes contaminem partições diferentes. As proporções podem variar dentro das tolerâncias definidas no YAML para preservar grupos.

## Regras metodológicas

- Ajustar encoders, normalizadores, projeções e agrupadores **somente no treino**.
- Usar validação para seleção de modelos, parâmetros e features.
- Manter teste isolado até a decisão final de configuração.
- Conferir presença das categorias e distribuição por split.
- Medir diferenças entre splits e investigar duplicatas ou sobreposições.
- Utilizar a mesma partição ao comparar ablações e arquiteturas.

## Variações e cuidados

Um `ticket_id` exclusivo não garante independência estatística entre registros. Chamados relacionados, imagens repetidas e textos quase idênticos ainda podem provocar contaminação. O agrupamento de imagens reduz esse risco, mas não resolve todos os tipos de duplicação.

Se houver alterações no dataset, os índices persistidos precisam ser validados ou regenerados mediante decisão explícita, com registro do experimento correspondente.

[Tratamento](TRATAMENTO_DE_DADOS.md) · [Avaliação](../analysis/AVALIACAO.md) · [Índice](../README.md)
