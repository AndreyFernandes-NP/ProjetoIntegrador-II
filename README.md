# Projeto Integrador II (PUC-SP)

Este repositório contém o desenvolvimento do Projeto Integrador II da Pós-Graduação em Inteligência Artificial da **PUC-SP**. O tema e objetivo deste projeto é desenvolver um sistema de análise automática de chamadas e evidências para uma empresa nacional de T.I, explorando todos os conceitos abordados durante o semestre na criação deste projeto.

## Objetivo

Desenvolver um sistema multimodal inteligente de Análise Automática de Chamados e Evidências para uma empresa corporativa de T.I, a triagem desses chamados atualmente é feita de forma manual por analistas especializados, é necessário desenvolver uma solução capaz de automatizar e manter a constiência entre cada resolução, possibilitando uma maior eficiência no atendimento ao cliente.

## Mapa de documentação do repositório

A ser criado.

## Escopo atual

Atualmente o foco do nosso projeto está, em ordem:

- limpeza, padronização e enriquecimento dos dados
- análise exploratória dos dados e indicadores de eficiência
- desenvolvimento dos primeiros testes de modelos de deep learning
- módulo de processamento de linguagem natural (PLN) para descrições textuais
- integração dos modelos multimodais
- consolidação final da pipeline e escalabilidade
- apresentação final e demo

## Requisitos

- Python 3.x
- `requirements.txt`
- Ambiente Virtual [Recomendado]

## Estrutura do repositório

```text
ProjetoIntegrador-II/
├── data/
├── docs/
├── notebooks/
├── src/
├── requirements.txt
├── README.md
└── LICENSE
```

## Setup local

1. **Clone o repositório**
   ```bash
   git clone https://github.com/AndreyFernandes-NP/ProjetoIntegrador-II.git
   cd ProjetoIntegrador-II
   ```
2. **Crie um virtual environment**
   ```bash
   python -m venv env
   ```
3. **Ative o ambiente virtual**
   ```bash
   env/Scripts/activate.bat
   ```
4. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```
5. **Execute nosso programa**
   ```bash
   python -m src
   ```

## Organização do projeto

A documentação e o desenvolvimento são organizados em frentes principais:

- `docs/`: documentação do projeto, dados, arquitetura, pipeline e análise
- `data/`: dados brutos, limpos, mapeamentos e bases processadas
- `src/`: código de pipeline, transformação, validação e modelos de ML
- `notebooks/`: análises exploratórias e visualizações
