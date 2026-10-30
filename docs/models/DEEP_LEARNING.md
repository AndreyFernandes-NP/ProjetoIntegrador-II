# Deep Learning Tabular

## Objetivo

Aprender representações de padrões recorrentes nos dados estruturados dos chamados e fornecer uma indicação compacta de perfil ao classificador multimodal.

## Abordagem de referência

A implementação discutida utiliza um autoencoder tabular seguido de agrupamento (*clustering*) da representação latente.

| Item | Referência |
| --- | --- |
| Arquivo | `src/models/profiles.py` |
| Campos numéricos | `sla_hours`, `sentiment_score`, `num_words` |
| Campos categóricos | `device_type`, `os`, `network_type` |
| Camada oculta | 32 unidades |
| Dimensão latente | 8 |
| Quantidade de clusters | 6 |
| Saída | `tabular_profile_cluster` |

## Interpretação do sinal

`tabular_profile_cluster` identifica um grupo de registros com características semelhantes. É uma categoria nominal e não uma escala ordinal de urgência, previsão de `target_category` ou uma pontuação contínua.

## Fluxo conceitual

1. Preparar campos numéricos e categóricos no treino.
2. Ajustar o autoencoder para obter uma representação compacta.
3. Agrupar os vetores latentes em perfis.
4. Aplicar as mesmas transformações aos demais splits.
5. Inserir o identificador de perfil na base enriquecida.

## Limitações e revisão

A escolha de seis clusters foi uma configuração experimental e não implica correspondência com as seis classes do alvo. Scores de separação modestos podem indicar grupos pouco distintos, a utilidade do perfil precisa ser confirmada pela melhora da classificação final em ablação.

[PLN](PLN.md) · [Multimodal](MULTIMODAL.md) · [Índice](../README.md)
