import json
from rag import load_corpus, get_embedding, cosine_similarity
from cache import save_cache
from evaluate import hit_rate, mrr, recall_at_k
from rank_bm25 import BM25Okapi


def build_bm25(texts):
    tokenized = [t.lower().split() for t in texts]
    return BM25Okapi(tokenized)


def bm25_ranked(bm25, question, k=20):
    scores = bm25.get_scores(question.lower().split())
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(score, i) for i, score in ranked[:k]]


def dense_ranked(question, corpus, k=20):
    q_emb = get_embedding(question)
    scored = []
    for i, c in enumerate(corpus):
        scored.append((cosine_similarity(q_emb, c["embedding"]), i))
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:k]


def rrf_fuse(dense, sparse, k=5, constant=60, dense_weight=0.5):
    sparse_weight = 1 - dense_weight
    scores = {}

    for rank, (score, i) in enumerate(dense, start=1):
        scores[i] = scores.get(i, 0) + dense_weight * (1 / (constant + rank))

    for rank, (score, i) in enumerate(sparse, start=1):
        scores[i] = scores.get(i, 0) + sparse_weight * (1 / (constant + rank))

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, i) for i, score in fused[:k]]


def run_hybrid(corpus, evalset, k=5, dense_weight=0.5):
    texts = [c["text"] for c in corpus]
    bm25 = build_bm25(texts)

    results = []
    for item in evalset:
        dense = dense_ranked(item["question"], corpus, k=20)
        sparse = bm25_ranked(bm25, item["question"], k=20)
        fused = rrf_fuse(dense, sparse, k=k, dense_weight=dense_weight)
        retrieved = [(score, corpus[i]) for score, i in fused]
        results.append((item["gold_ids"], retrieved))
    save_cache()
    return results


if __name__ == "__main__":
    corpus = load_corpus()
    with open("evalset.json") as f:
        evalset = json.load(f)

    print(f"corpus: {len(corpus)} passages, {len(evalset)} questions")
    print()

    for w in [1.0, 0.7, 0.5, 0.3, 0.0]:
        res = run_hybrid(corpus, evalset, k=5, dense_weight=w)
        if w == 1.0:
            label = "dense only"
        elif w == 0.0:
            label = "bm25 only"
        else:
            label = f"hybrid w={w}"
        print(f"{label:<14} hit {hit_rate(res):.3f}  mrr {mrr(res):.3f}  recall {recall_at_k(res):.3f}")