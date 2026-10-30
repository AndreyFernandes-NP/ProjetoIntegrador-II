# Fontes de Dados

## Base de chamados

O projeto utiliza uma base de chamados sintética de suporte corporativo de TI. A etapa de processamento toma como referência um CSV tratado, configurado em `src/config/sources.yaml`:

- **Entrada de referência:** `data/clean/chamados-clean.csv`.
- **Saída processada:** `data/processed/chamados.parquet`.
- **Unidade de análise:** um chamado por `ticket_id`.
- **Alvo de classificação:** `target_category`.

## Modalidades utilizadas

| Modalidade | Campos típicos | Finalidade |
| --- | --- | --- |
| Texto | `title`, `description` | Relato e intenção do solicitante |
| Imagem | `image_path` | Evidências anexadas ao atendimento |
| Tabular | `sla_hours`, `device_type`, `os`, `network_type`, `num_words`, entre outros | Contexto e características estruturadas |
| Rótulo | `target_category` | Referência para treino e avaliação do classificador final |

## Evidências visuais e anotações

Os caminhos em `image_path` identificam arquivos visuais, quando existe treinamento orientado de geração de legendas, as anotações podem ser armazenadas em `data/annotations/image_captions.csv`. A existência de um caminho não garante que o arquivo esteja presente ou seja uma evidência distinta.

[Dicionário](DICIONARIO_DE_DADOS.md) · [Tratamento](TRATAMENTO_DE_DADOS.md) · [Índice](../README.md)
