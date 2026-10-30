# Projeto Integrador II (PUC-SP)

Esse repositório reúne o desenvolvimento do **Projeto Integrador II da Pós-Graduação em Inteligência Artificial da PUC-SP**. O projeto investiga a classificação automática de chamados de suporte de tecnologia da informação, combinando informações textuais, visuais e tabulares em uma abordagem multimodal.

> **Estado do projeto:** desenvolvimento e avaliação. A documentação descreve a arquitetura adotada e as frentes de trabalho em evolução, os resultados e os procedimentos operacionais estão sendo consolidados e serão atualizados ao decorrer dos últimos momentos.

## Objetivo

Desenvolver e avaliar um sistema de triagem inteligente de chamados de TI, capaz de utilizar o título, a descrição, evidências visuais e os atributos estruturados de cada solicitação para apoiar a classificação de sua categoria. A proposta busca reduzir a dependência de triagem exclusivamente manual e examinar a contribuição de diferentes modalidades de dados para a consistência das decisões.

A variável-alvo do estudo é `target_category`, com seis categorias de referência: **Software, Rede, Hardware, RH, Acesso e Financeiro**.

## Mapa de documentação do repositório

| Área | Documentação | Conteúdo |
| --- | --- | --- |
| **Índice geral** | [Documentação (`docs/`)](docs/README.md) | Navegação completa pelos documentos |
| **Planejamento** | [Escopo](docs/project/ESCOPO.md) · [Objetivos](docs/project/OBJETIVOS.md) · [Etapas](docs/project/ETAPAS.md) · [Decisões](docs/project/DECISOES.md) | Problema, entregáveis e diretrizes |
| **Dados** | [Fontes](docs/data/FONTES_DE_DADOS.md) · [Dicionário](docs/data/DICIONARIO_DE_DADOS.md) · [Tratamento](docs/data/TRATAMENTO_DE_DADOS.md) · [Divisão e validação](docs/data/DIVISAO_E_VALIDACAO.md) | Dados, transformações e qualidade |
| **Arquitetura** | [Arquitetura](docs/architecture/ARQUITETURA.md) · [Pastas](docs/architecture/ESTRUTURA_DE_PASTAS.md) · [Pipeline](docs/architecture/PIPELINE.md) · [Configurações](docs/architecture/CONFIGURACOES.md) | Organização e fluxo da solução |
| **Modelos** | [Visão](docs/models/VISAO_COMPUTACIONAL.md) · [PLN](docs/models/PLN.md) · [Deep Learning](docs/models/DEEP_LEARNING.md) · [Multimodal](docs/models/MULTIMODAL.md) · [Contratos de features](docs/models/CONTRATOS_DE_FEATURES.md) | Geração de sinais e classificação |
| **Análise** | [Metodologia](docs/analysis/METODOLOGIA.md) · [Hipóteses](docs/analysis/HIPOTESES.md) · [Avaliação](docs/analysis/AVALIACAO.md) · [Experimentos e limitações](docs/analysis/EXPERIMENTOS_E_LIMITACOES.md) | Método e interpretação dos resultados |

## Organização do repositório

```text
ProjetoIntegrador-II/
├── data/                  # Bases e arquivos auxiliares de dados
│   ├── clean/             # Entrada tratada utilizada pelo processamento
│   ├── processed/         # Parquets e índices de divisão
│   └── annotations/       # Legendas e anotações, quando disponíveis
├── docs/                  # Documentação acadêmica e técnica
│   ├── project/           # Escopo, objetivos, etapas e decisões
│   ├── data/              # Fontes, dicionário, tratamento e validações
│   ├── architecture/      # Estrutura, configurações e pipeline
│   ├── models/            # Visão, PLN, deep learning e fusão
│   └── analysis/          # Metodologia, hipóteses e avaliação
├── notebooks/             # Exploração de dados e experimentação
├── src/
│   ├── config/            # Arquivos YAML e configurações compartilhadas
│   ├── data/              # Processamento, validação e divisão
│   └── models/            # Geradores, treinamento e fusão multimodal
├── reports/               # Relatórios e artefatos de execuções
├── requirements.txt       # Dependências Python
├── LICENSE
└── README.md
```

A árvore acima representa a organização de referência da versão em desenvolvimento. Algumas subpastas são criadas na geração de artefatos e podem não estar presentes ou versionadas em todas as revisões do repositório.

Também há uma explicação dedicada de cada diretório em [Estrutura de Pastas](docs/architecture/ESTRUTURA_DE_PASTAS.md), além dos índices locais [data/](data/README.md), [src/](src/README.md), [notebooks/](notebooks/README.md) e [reports/](reports/README.md).

## Abordagem adotada

A solução é dividida em três geradores independentes de características — **visão computacional**, **processamento de linguagem natural** e **deep learning tabular**. As informações produzidas por essas etapas são combinadas com variáveis originais em uma base enriquecida. Um classificador multimodal posterior é responsável pela predição final.

Os geradores não recebem `target_category` como entrada e não são tratados como classificadores finais de chamados. A seleção das novas features permanece sujeita a testes.

## Escopo atual

- Padronização, qualidade e rastreabilidade dos dados.
- Desenvolvimento e revisão dos geradores de características para as três modalidades.
- Seleção de sinais compactos e relevantes para a classificação multimodal.
- Comparação de modelos, análise de generalização e prevenção de vazamento de informações.
- Consolidação da documentação acadêmica e técnica.

## Execução e instalação

**A documentar.** O sistema ainda não está completamente pronto e serão necessários modificações que podem alterar a ordem de execução dos arquivos do projeto ou simplificá-los.

1. **Clone o repositório**
   ```bash
   git clone https://github.com/AndreyFernandes-NP/ProjetoIntegrador-II.git
   cd ProjetoIntegrador-II
   ```
2. **Crie um virtual environment**
   ```bash
   python -m venv .venv
   ```
3. **Ative o ambiente virtual**
   ```bash
   .venv/Scripts/Activate.ps1
   ```
4. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```

## Referência

Esse projeto dá continuidade ao trabalho acadêmico e ao padrão documental do [Projeto Integrador I](https://github.com/AndreyFernandes-NP/ProjetoIntegrador-I), mantendo a separação entre escopo, dados, arquitetura e metodologia, agora com uma área específica para os modelos multimodais.
