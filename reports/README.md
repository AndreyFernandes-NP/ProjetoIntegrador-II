# Relatórios (`reports/`)

Destinado ao registro de métricas, configurações, previsões e artefatos de execução, permitindo comparar versões e sustentar as conclusões do estudo.

## Organização de referência

- `reports/data/`: relatórios de qualidade dos dados, quando gerados.
- `reports/runs/<execucao>/`: registros de experimentos e enriquecimento.
- `reports/runs/<execucao>/enriched.parquet`: dataset enriquecido, conforme etapa e revisão.
- Metadados como `run.json`, `feature_schema.json` e `stage_metrics.json` podem acompanhar as execuções.

O armazenamento dos pesos pode ocorrer separadamente em `models/runs/<execucao>/`.

## Interpretação

Todo resultado precisa estar associado ao dataset, split, seed, configuração e versão dos modelos que o produziram. Não comparar pontuações de versões distintas como se fossem o mesmo experimento nem divulgar métricas de validação como resultados finais de teste.

[Avaliação](../docs/analysis/AVALIACAO.md) · [Experimentos](../docs/analysis/EXPERIMENTOS_E_LIMITACOES.md) · [README principal](../README.md)
