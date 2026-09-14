# manual-rag-engine

A RAG pipeline built without frameworks to understand how retrieval actually works, then upgraded with production retrieval patterns and measured against a labelled dataset.

The embedding comparison maths is written by hand. No LangChain, no vector database library.

## What it does

Loads a corpus of biomedical passages, embeds each one, and stores the vectors in memory. A question comes in, gets embedded, and is compared against every passage using cosine similarity. The top k passages are passed to the model as context with a system prompt telling it to answer only from what it was given.

Retrieval supports metadata filtering applied before scoring, and an optional hybrid mode that fuses dense similarity with BM25 keyword matching using reciprocal rank fusion.

## Data

BioASQ, sampled down to 2,316 passages and 100 questions. 816 of those passages answer at least one question, the other 1,500 are distractors so retrieval has to actually discriminate rather than pick from a pool where everything is correct.

Each question carries the IDs of the passages that answer it, so scoring is exact rather than heuristic. Questions average around 8 relevant passages each.

## Results

### Top-k

| k | hit rate | mrr | recall |
|---|---|---|---|
| 1 | 0.890 | 0.890 | 0.290 |
| 3 | 0.900 | 0.895 | 0.481 |
| 5 | 0.910 | 0.897 | 0.589 |

Hit rate barely moves from k=1 to k=5. Recall doubles.

That gap is the finding. With around 8 gold passages per question, top-1 retrieval was structurally incapable of gathering what a question needed, no matter how good the ranking was. Hit rate said the upgrade was worth 2 percent. Recall said it was worth doubling coverage.

At k=5 a question needing 8 passages caps out at 0.625 recall, so 0.589 is close to the ceiling for that k.

### Hybrid search

| configuration | hit rate | mrr | recall |
|---|---|---|---|
| dense only | 0.910 | 0.897 | 0.589 |
| hybrid, dense weight 0.7 | 0.930 | 0.885 | 0.584 |
| hybrid, dense weight 0.5 | 0.920 | 0.888 | 0.559 |
| hybrid, dense weight 0.3 | 0.840 | 0.809 | 0.493 |
| bm25 only | 0.810 | 0.747 | 0.438 |

BM25 at low weight acts as a safety net for rare exact terms rather than a general improvement. It rescues a couple of questions dense retrieval misses entirely, at a small cost to ranking elsewhere.

The failure it fixes is visible in a single example. Asked "is capmatinib effective for glioblastoma", dense retrieval returned five passages about glioblastoma generally and none about capmatinib. The embedding latched onto the common, heavily represented term and lost the rare drug name. BM25 weights rare terms heavily, which is exactly the gap.

Which configuration wins depends on what you are optimising. Dense weight 0.7 has the best hit rate. Dense only has the best MRR and recall.

### Generation quality (RAGAS)

| metric | score |
|---|---|
| faithfulness | 0.842 |
| answer relevancy | 0.685 |
| context precision | 0.725 |

Faithfulness measures whether the answer is grounded in the retrieved context rather than invented. 0.842 means most claims trace back to what was retrieved.

Answer relevancy is the lowest score and it is partly measuring the system working correctly. When retrieval misses, the system prompt makes the model say it does not know. That is the right behaviour, but a relevancy judge scores a refusal near zero. So some of that 0.685 is honest refusals rather than bad generation.

Context precision cross-checks the hand-rolled retrieval metrics from a different angle, using the reference answer rather than passage IDs.

### Metadata filtering

Filtering narrows the candidate pool before any similarity is computed, not after. Filtering after top-k would mean asking for 5 results and getting 2.

On a question where the top results were already relevant, filtering to relevant passages changed nothing. Forced to search only distractors, scores dropped from 0.581 to 0.506 and it returned different, genuinely unrelated passages.

That is the honest claim: filtering enforces scope rather than improving relevance. Its value is that a query cannot reach passages it should not see, which is what makes RAG viable with permissions or date boundaries.

## Files

- `rag.py` — embeddings, cosine similarity, retrieval with filtering, generation
- `cache.py` — disk-backed embedding cache keyed by md5 of the text
- `build_corpus.py` — samples BioASQ down to a workable corpus and eval set
- `evaluate.py` — hit rate, MRR, recall at various k
- `hybrid.py` — BM25, RRF, and the fusion weight sweep
- `filtering.py` — metadata filtering demonstration
- `ragas_eval.py` — faithfulness, answer relevancy, context precision
- `main.py` — FastAPI endpoint

## API

```
uvicorn main:app --reload
```

POST to `/ask`:

```json
{"query": "is capmatinib effective for glioblastoma?", "k": 5, "relevant_only": false}
```

Returns the answer plus every source passage with its ID and similarity score. Returning the sources matters. When an answer looks wrong you can immediately tell whether retrieval fetched the wrong passages or the model misused the right ones. Two different bugs with two different fixes.

## Running it

```
pip install -r requirements.txt
python build_corpus.py
python evaluate.py
```

Needs `OPENAI_API_KEY` in `.env`.

First run embeds 2,316 passages, which takes a few minutes and costs a couple of cents. After that everything is cached.

## Limitations

- Vectors live in a Python list, so retrieval is a linear scan against all 2,316 passages on every query. Fine at this size, unworkable past a few thousand. Real vector databases use approximate nearest neighbour indexes that trade a little recall for a large speed gain.
- 100 questions is small. Differences under about 0.05 are probably noise.
- Cosine similarity is hand-written without numpy. Deliberate, for understanding, but slow.
- The `relevant` metadata flag is synthetic. It marks passages that answer an eval question, which is not a real-world attribute, so it demonstrates the filtering mechanism without being able to show a relevance gain.
- RAGAS metrics were computed at k=3, not k=5. Faithfulness verdict generation exceeded the token limit at five passages of biomedical text, so those scores do not describe the same retrieval configuration as the hit rate and MRR numbers.

## Next

Reranking with a cross-encoder, a hosted vector store, LangSmith tracing, and containerised deployment.

## A note on the cross-corpus result

The same hybrid search code was run on a different corpus in a separate project (conversational financial Q&A) where it reduced both hit rate and MRR at every fusion weight, the opposite of the result here.

Biomedical text is full of rare, distinguishing tokens like drug and gene names, which is what BM25 is good at. The financial corpus shared vocabulary like "account" and "money" across almost every document, so keyword matching surfaced topically wrong passages with high confidence.

Hybrid search is corpus-dependent. Neither result generalises, which is the argument for measuring it on your own data rather than following the default advice.