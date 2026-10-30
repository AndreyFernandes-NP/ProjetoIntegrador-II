# Dicionário de Dados

Esse dicionário reúne os campos essenciais para compreender o pipeline. A relação completa e os tipos finais devem ser verificados no `sources.yaml` da revisão efetivamente utilizada.

## Campos de identificação e contexto

| Coluna | Papel | Observação |
| --- | --- | --- |
| `ticket_id` | Identificador único do chamado | É a nossa chave de rastreabilidade |
| `title` | Título do chamado | Entrada textual de PLN |
| `description` | Descrição detalhada | Entrada textual de PLN |
| `image_path` | Caminho de imagem | Entrada da visão e agrupamento de splits |
| `target_category` | Categoria real | Alvo exclusivo do classificador final |

## Campos estruturados relevantes

| Coluna | Significado operacional | Cuidado |
| --- | --- | --- |
| `sla_hours` | Prazo de atendimento em horas | Não deve carregar informação posterior à decisão de triagem |
| `sentiment_score` | Pontuação de sentimento | Recalculada pelo gerador de PLN pra garantir confiabilidade |
| `num_words` | Quantidade de palavras | Indicador derivado do relato |
| `device_type` | Tipo de dispositivo | Categoria tabular |
| `os` | Sistema operacional | Categoria tabular |
| `network_type` | Tipo de rede | Categoria tabular |
| `resolution_time_hours` | Tempo até resolução | Não usar para prever a categoria na abertura, se só for conhecido depois |

## Sinais enriquecidos em estudo

| Coluna / família | Origem | Interpretação |
| --- | --- | --- |
| `image_description` | Visão | Texto descritivo gerado sobre a imagem |
| `image_score` | Visão | Score associado à geração da legenda, não à correção da categoria |
| `sentiment_score` (recalculado) | PLN | Polaridade estimada do relato |
| `main_topic` | PLN — versão anterior | Tópico inferido atualmente sob revisão se deve ou não continuar |
| `semantic_margin` | PLN — versão anterior | Diferença entre os dois principais scores de tópico |
| `urgency_score` | PLN — versão anterior | Urgência inferida, utilidade ainda em avaliação |
| Sinais de intenção (nomes a definir) | PLN — alternativa experimental | Evidências de falha, acesso, orientação e lentidão |
| `tabular_profile_cluster` | Deep learning | Identificador nominal de perfil tabular |

## Categorias do alvo

`Software`, `Rede`, `Hardware`, `RH`, `Acesso` e `Financeiro`.

[Fontes](FONTES_DE_DADOS.md) · [Divisão e validação](DIVISAO_E_VALIDACAO.md) · [Contratos](../models/CONTRATOS_DE_FEATURES.md) · [Índice](../README.md)
