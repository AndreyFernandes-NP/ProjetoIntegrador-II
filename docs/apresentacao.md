# Planejamento da Apresentação — Banca PUC-SP

> Rascunho de roteiro para a apresentação do Projeto Integrador II. Ir preenchendo/ajustando conforme o projeto evolui.

## 1. A história que queremos contar

A banca não quer só ver gráficos e código — quer entender uma jornada: **qual era o problema, o que encontramos pelo caminho, que decisões tomamos e por quê, e onde isso nos deixa agora.**

Arco sugerido:

```
Problema real  →  Primeiro contato com os dados  →  Obstáculo (dados quebrados)
    →  Diagnóstico  →  Decisão técnica  →  Resultado  →  Próximos passos
```

Cada bloco abaixo vira um bloco de slide(s).

---

## 2. Estrutura sugerida dos slides

### Bloco 1 — Contexto e problema (2-3 slides)
- Empresa nacional de T.I. faz triagem de chamados **manualmente**, por analistas especializados.
- Consequência: lentidão, inconsistência entre resoluções, custo de operação alto.
- Pergunta que guia o projeto: *"Dá pra automatizar essa triagem de forma multimodal (texto + imagem + dados estruturados) mantendo consistência?"*

### Bloco 2 — Objetivo do projeto (1 slide)
- Sistema multimodal inteligente de Análise Automática de Chamados e Evidências.
- Deixar claro que é um projeto em **etapas** (não é "treinamos um modelo e pronto") — mostra maturidade de processo pra banca.

### Bloco 3 — Os dados que recebemos (1-2 slides)
- ~5.000 chamados, 44 colunas: dados estruturados (device, SO, prioridade, SLA...), texto (título/descrição) e imagem (anexo do chamado).
- Aqui entra o **gancho de storytelling**: "só que quando fomos explorar os dados, percebemos que várias colunas não faziam sentido."

### Bloco 4 — O obstáculo: dados de um merge quebrado (2 slides) ⭐ ponto alto da história
- Ao investigar, descobrimos que o dataset bruto vinha de um `merge` entre chamados e uma base de empresas/eventos que **quase não deu match**:
  - `company`, `segment`, `vip`, `monthly_revenue` etc. — presentes em só **2 de 5.000 linhas** (99,96% vazias).
  - `num_events`, `num_abertura`, `num_updates`, `num_encerramento` — sempre **zero**, sem informação nenhuma.
  - `city_x`/`city_y`, `state_x`/`state_y`, `num_images_x`/`num_images_y` — colunas duplicadas pelo próprio merge, uma delas sempre vazia.
- Mostrar um print/tabela pequena com o "antes" ajuda muito aqui — é visual e concreto.

### Bloco 5 — Decisão técnica: cleaner genérico, não gambiarra pontual (2 slides)
- Em vez de tratar isso na mão (o que resolveria só *esse* CSV), construímos um **cleaner genérico** (`src/data/cleaner.py`) com regras reaproveitáveis para qualquer DataFrame da pipeline:
  1. `merge_suffix_pairs` — funde colunas duplicadas de merge (`_x`/`_y`) no nome base, sem hardcode de nome de coluna.
  2. `drop_sparse_columns` — remove colunas praticamente vazias (threshold configurável, ex. ≥99% nulo).
  3. `drop_constant_columns` — remove colunas sem variação (sempre o mesmo valor).
  4. Inferência automática de tipo (número/data) e normalização de nomes de coluna/acentuação.
- **Por que isso importa pra apresentação**: mostra que a decisão foi de engenharia, pensando em reuso pro resto do projeto (as próximas bases que entrarem na pipeline passam pelo mesmo cleaner).

### Bloco 6 — Resultado (1-2 slides)
- Antes → depois:

| | Bruto | Limpo |
|---|---|---|
| Colunas | 44 | 28 |
| Colunas 100%/quase 100% vazias | 12 | 0 |
| Valores nulos remanescentes | — | 0 |
| Linhas | 5.000 | 5.000 |

- Dataset final: estruturado, sem ruído de merge, pronto pra virar insumo de EDA e dos primeiros modelos.

### Bloco 7 — Onde estamos no roadmap (1 slide)
Usar o roadmap do README como linha do tempo, marcando o que já foi feito:

- [x] Limpeza, padronização e enriquecimento dos dados
- [ ] Análise exploratória dos dados e indicadores de eficiência
- [ ] Primeiros testes de modelos de deep learning
- [ ] Módulo de PLN para descrições textuais
- [ ] Integração dos modelos multimodais
- [ ] Consolidação final da pipeline e escalabilidade
- [ ] Apresentação final e demo

### Bloco 8 — Próximos passos concretos (1 slide)
- EDA: distribuição de categorias (`target_category`), prioridade x tempo de resolução, SLA x sentimento.
- Definir baseline (ex.: classificador simples texto→categoria) antes de partir pra multimodal.

---

## 3. Perguntas prováveis da banca (preparar respostas)

- **"Por que os dados vieram tão sujos assim?"** — simular um cenário real de empresa (dados vêm de sistemas legados/joins malfeitos); parte do desafio do projeto é lidar com isso.
- **"Por que não usar essas colunas de empresa mesmo com poucos dados?"** — 2 em 5.000 não é estatisticamente utilizável; manter geraria uma feature quase toda nula, sem ganho.
- **"O cleaner é específico desse dataset ou reutilizável?"** — reutilizável: nenhuma regra tem nome de coluna fixo, é tudo baseado em thresholds/estrutura.
- **"O que muda se a próxima base de dados vier diferente?"** — o cleaner já lida com merges malformados e colunas esparsas/constantes automaticamente.

---

## 4. Pendências deste documento

- [ ] Definir se vai ter demo ao vivo ou só slides.
- [ ] Escolher 2-3 exemplos de chamados (com imagem) pra ilustrar o dado bruto.
- [ ] Adicionar gráficos da EDA assim que existirem (`notebooks/`).
- [ ] Definir quem fala qual bloco (se apresentação em grupo).
- [ ] Tempo total da apresentação e tempo por bloco.



SIM USEI UMA IA PRA MONTAR A ESTRUTURA SÓ, DEPOIS EU REVISO E 
PLANEJO TUDO, NAO ME JULGUEM PFV :)