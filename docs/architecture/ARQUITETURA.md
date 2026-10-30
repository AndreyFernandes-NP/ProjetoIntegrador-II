# Arquitetura do Projeto

## Visão geral

A solução segue uma arquitetura modular, organizada em preparação de dados, geração de características por modalidade, consolidação da base enriquecida, classificação multimodal e avaliação.

```text
Chamados (texto + imagem + dados tabulares)
                |
         Processar / validar
                |
       Partições reproduzíveis
                |
     +----------+-----------+
     |          |           |
    Visão       PLN      Deep learning
     |          |           |
     +----------+-----------+
                |
       Features enriquecidas
                |
        Fusão multimodal
                |
        Categoria prevista
                |
        Métricas / relatórios
```

O diagrama é conceitual: a implementação das etapas pode ser sequencial para satisfazer dependências, por exemplo, quando o modelo tabular consome um sentimento recalculado pelo PLN. Resumidamente, os modelos podem interagir entre si e também utilizar-se dos dados gerados.

## Camadas principais

### Dados

`src/data/` concentra leitura, transformação, validação e divisão. Os artefatos são organizados em `data/` e, para experimentos, em diretórios de relatório.

### Configuração

`src/config/sources.yaml` descreve as regras de dados; `src/config/models.yaml` descreve os geradores, fontes pré-treinadas, hiperparâmetros e as saídas esperadas.

### Geradores

`src/models/` contém o código das modalidades. A visão utiliza uma arquitetura de captioning, o PLN interpreta título e descrição e o deep learning cria perfis tabulares. As saídas devem respeitar contratos e tipos definidos na configuração.

### Integração e avaliação

O enriquecimento cria uma tabela única com as features adicionais. A fusão aprende a categoria final utilizando os sinais permitidos e produz relatórios de avaliação.

## Princípios arquiteturais

- **Modularidade:** cada modelo tem responsabilidade e saídas identificáveis.
- **Configurabilidade:** evitar parâmetros operacionais escondidos no código.
- **Reprodutibilidade:** registrar seed, configuração, splits, artefatos e métricas.
- **Sem vazamento:** o alvo final não é uma entrada dos geradores.
- **Evolução controlada:** features experimentais podem ser substituídas sem alterar a estrutura geral.

[Estrutura](ESTRUTURA_DE_PASTAS.md) · [Pipeline](PIPELINE.md) · [Configurações](CONFIGURACOES.md) · [Índice](../README.md)
