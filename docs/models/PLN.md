# Processamento de Linguagem Natural (PLN)

## Objetivo e implementação

Utilizar conjuntamente `title` e `description` para produzir sinais compactos que ajudem o multimodal. A etapa não recebe `target_category` e não escolhe a categoria final do chamado.

A configuração principal usa `adapter: semantic_nlp`, implementado em `src/models/language.py`, com o encoder de `src/models/semantic_text.py`. As duas redes permanecem congeladas:

- `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` mantém o cálculo de sentimento por hipóteses NLI.
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` representa os textos e os exemplos operacionais.

## Saídas atuais

| Coluna | Interpretação | Ação |
| --- | --- | --- |
| `sentiment_score` | Polaridade estimada pelas hipóteses de sentimento | Substituir |
| `text_failure_similarity` | Proximidade a relatos de falha ou indisponibilidade | Adicionar |
| `text_access_request_similarity` | Proximidade a pedidos de conta, senha ou permissão | Adicionar |
| `text_guidance_similarity` | Proximidade a dúvidas e pedidos de orientação | Adicionar |
| `text_performance_similarity` | Proximidade a relatos de lentidão ou travamentos | Adicionar |

Todas são numéricas em `[-1, 1]`. As quatro similaridades são cossenos, não probabilidades ou confiança calibrada. Podem ser elevadas simultaneamente, não somam necessariamente 1 e não existe escolha obrigatória de uma intenção. Um valor baixo indica pouca proximidade aos exemplos nas configurações.

## Cálculo das similaridades

1. Normalizar espaços e codificar textos distintos, evitando repetir a mesma inferência.
2. Codificar título e descrição separadamente, com média dos tokens que não são padding e normalização dos vetores.
3. Somar os dois vetores com pesos iguais e normalizar novamente. Se um campo estiver vazio, utilizar o outro; rejeitar chamados com ambos vazios.
4. Codificar os exemplos de cada grupo em `modelos.pln.semantic.prototypes`.
5. Retornar, para cada grupo, o maior cosseno entre o vetor do chamado e seus exemplos.

O encoder gera 384 dimensões internas, o limite padrão é de 128 tokens por campo, portanto textos mais longos são truncados. O sentimento preserva a composição textual e o limite de 256 tokens da implementação anterior.

Os exemplos são descrições operacionais, não palavras-chave obrigatórias nem rótulos supervisionados das seis categorias finais. Não há fine-tuning, ajuste de protótipos ou leitura do target nessa etapa.

## Configuração e persistência

`semantic.fonte` define nome/link e revisão do encoder; `batch_size` controla o lote e `max_length` o truncamento. O adaptador exige um encoder compatível com média mascarada dos tokens. Trocar para uma arquitetura que exija outro pooling demanda adaptar o código.

Os nomes dos grupos definem as colunas. Ao incluir, remover ou renomear grupos, sincronizar `pln.outputs`, `multimodal.features.numeric` e `enrichment.max_new_columns`. Os outputs exigem `dtype: float`, `action: add` e `range: [-1, 1]`. O sentimento continua sendo a única substituição permitida.

Downloads ficam em `models/huggingface/`. Cada execução salva as duas redes, tokenizers e `semantic_prototypes.npz` em `models/runs/<execução>/pln/`. A inferência restaura esses arquivos locais com a configuração original, sem download ou ajuste. O manifesto de hashes da execução também cobre os artefatos semânticos.

Referência do encoder e do pooling: [model card oficial](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).

[Contratos](CONTRATOS_DE_FEATURES.md) · [Experimentos](../analysis/EXPERIMENTOS_E_LIMITACOES.md) · [Índice](../README.md)
