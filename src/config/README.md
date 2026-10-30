# Configurações (`src/config/`)

Concentra definições que controlam o processamento de dados e as etapas de modelagem, evitando parâmetros espalhados pelo código.

## Arquivos principais

- `sources.yaml`: entrada e saída do dataset, conversões, descarte de colunas e regras de qualidade.
- `models.yaml`: modelos habilitados, fontes pré-treinadas, features de entrada, treinamento e contratos de saída.
- Outros arquivos de configuração ou constantes: conferir a revisão em uso.

Qualquer mudança de nome ou tipo de feature deve ser refletida no gerador, nas validações e no pipeline multimodal. Parâmetros de experimentos anteriores precisam permanecer rastreáveis para comparação justa.

[Configurações detalhadas](../../docs/architecture/CONFIGURACOES.md) · [Contratos de features](../../docs/models/CONTRATOS_DE_FEATURES.md) · [README principal](../../README.md)
