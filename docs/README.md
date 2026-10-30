# Documentação do Projeto Integrador II

Essa pasta centraliza as definições e técnicas do projeto da análise automática de chamados e evidências de TI. O objetivo é explicar o problema estudado, a origem dos dados, a estrutura do código, as decisões de modelagem e os critérios de avaliação.

Os documentos descrevem tanto componentes já trabalhados quanto pontos sob revisão.

## Estrutura da documentação

```text
docs/
├── project/          # Planejamento e decisões
├── data/             # Dados e qualidade
├── architecture/     # Componentes e fluxo do sistema
├── models/           # Modelos e sinais multimodais
├── analysis/         # Hipóteses, experimentos e avaliação
└── README.md         # Índice atual
```

## Mapa de navegação

### Planejamento — `project/`
- [Escopo](project/ESCOPO.md): problema, limites e entregas previstas.
- [Objetivos](project/OBJETIVOS.md): objetivos gerais e específicos.
- [Etapas](project/ETAPAS.md): organização da evolução do projeto.
- [Decisões](project/DECISOES.md): princípios de arquitetura e decisões revisáveis.

### Dados — `data/`
- [Fontes de dados](data/FONTES_DE_DADOS.md): origem e natureza dos chamados.
- [Dicionário de dados](data/DICIONARIO_DE_DADOS.md): campos relevantes e seu papel.
- [Tratamento de dados](data/TRATAMENTO_DE_DADOS.md): limpeza e transformação.
- [Divisão e validação](data/DIVISAO_E_VALIDACAO.md): particionamento, qualidade e vazamento.

### Arquitetura — `architecture/`
- [Arquitetura](architecture/ARQUITETURA.md): componentes e responsabilidades.
- [Estrutura de pastas](architecture/ESTRUTURA_DE_PASTAS.md): localização dos arquivos.
- [Pipeline](architecture/PIPELINE.md): jornada dos dados e dos modelos.
- [Configurações](architecture/CONFIGURACOES.md): papel dos arquivos declarativos.

### Modelos — `models/`
- [Visão computacional](models/VISAO_COMPUTACIONAL.md): interpretação das evidências visuais.
- [PLN](models/PLN.md): sinais derivados de título e descrição.
- [Deep learning](models/DEEP_LEARNING.md): perfis tabulares.
- [Modelo multimodal](models/MULTIMODAL.md): integração das modalidades.
- [Contratos das features](models/CONTRATOS_DE_FEATURES.md): entradas, saídas e controles.

### Análise — `analysis/`
- [Metodologia](analysis/METODOLOGIA.md): estratégia de pesquisa.
- [Hipóteses](analysis/HIPOTESES.md): hipóteses testáveis.
- [Avaliação](analysis/AVALIACAO.md): métricas, splits e comparação.
- [Experimentos e limitações](analysis/EXPERIMENTOS_E_LIMITACOES.md): histórico e cautelas.

## Documentação por diretório do código

Os arquivos [data/README.md](../data/README.md), [src/README.md](../src/README.md), [src/config/README.md](../src/config/README.md), [src/data/README.md](../src/data/README.md), [src/models/README.md](../src/models/README.md), [notebooks/README.md](../notebooks/README.md) e [reports/README.md](../reports/README.md) descrevem a finalidade de cada pasta próxima de seus arquivos.

[Voltar ao README principal](../README.md)
