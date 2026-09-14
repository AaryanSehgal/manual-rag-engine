import json
from rag import load_corpus, retrieve, get_embedding
from cache import save_cache


def hit_rate(results):
    hits = 0
    for gold_ids, retrieved in results:
        found = [c["id"] for score, c in retrieved]
        if any(g in found for g in gold_ids):
            hits += 1
    return hits / len(results)


def mrr(results):
    total = 0
    for gold_ids, retrieved in results:
        for rank, (score, c) in enumerate(retrieved, start=1):
            if c["id"] in gold_ids:
                total += 1 / rank
                break
    return total / len(results)


def recall_at_k(results):
    recalls = []
    for gold_ids, retrieved in results:
        found = [c["id"] for score, c in retrieved]
        got = len([g for g in gold_ids if g in found])
        recalls.append(got / len(gold_ids))
    return sum(recalls) / len(recalls)


def run(corpus, evalset, k=5, filters=None):
    results = []
    for item in evalset:
        retrieved = retrieve(item["question"], corpus, k=k, filters=filters)
        results.append((item["gold_ids"], retrieved))
    save_cache()
    return results


if __name__ == "__main__":
    corpus = load_corpus()
    with open("evalset.json") as f:
        evalset = json.load(f)

    print(f"corpus: {len(corpus)} passages, {len(evalset)} questions")
    print()

    for k in [1, 3, 5, 10]:
        res = run(corpus, evalset, k=k)
        print(f"k={k:<3} hit {hit_rate(res):.3f}  mrr {mrr(res):.3f}  recall {recall_at_k(res):.3f}")
        