# Escopo do Projeto

## Contexto e problema

A triagem de chamados de suporte de TI costuma demandar leitura de relatos e inspeção de evidências por analistas. O projeto investiga uma solução de apoio à classificação capaz de explorar texto, imagens e atributos estruturados do atendimento.

## Recorte

- Unidade de observação: chamado identificado por `ticket_id`.
- Variável-alvo: `target_category`.
- Categorias de referência: Software, Rede, Hardware, RH, Acesso e Financeiro.
- Modalidades: conteúdo de `title` e `description`, imagens associadas em `image_path` e campos tabulares disponíveis no momento da triagem.
- Produto acadêmico: pipeline e avaliação experimental de um classificador multimodal.

## Dentro do escopo

- Preparar uma base padronizada com validações declarativas.
- Implementar partições reproduzíveis e controles contra vazamento entre amostras.
- Gerar sinais textuais, visuais e tabulares de forma modular.
- Consolidar uma base enriquecida com poucas features adicionais.
- Treinar, comparar e avaliar classificadores de fusão multimodal.
- Registrar métricas, configurações, artefatos e limitações.

## Fora do escopo atual

- Automatizar a resolução ou o encerramento do chamado.
- Substituir o julgamento humano em decisões críticas de atendimento.
- Prometer integração com sistemas corporativos reais ou implantação em produção.
- Tratar o reconhecimento de categoria por um gerador isolado como solução multimodal final.

## Documentos relacionados

[Objetivos](OBJETIVOS.md) · [Etapas](ETAPAS.md) · [Arquitetura](../architecture/ARQUITETURA.md) · [Avaliação](../analysis/AVALIACAO.md) · [Índice](../README.md)
