import json
from rag import load_corpus, retrieve, generate_answer

print("embedding corpus, this takes a few minutes first time...")
corpus = load_corpus()
print("corpus loaded:", len(corpus))

with open("evalset.json") as f:
    evalset = json.load(f)

q = evalset[0]
print("\nquestion:", q["question"])
print("gold ids:", q["gold_ids"])

hits = retrieve(q["question"], corpus, k=5)
print("\nretrieved:")
for score, c in hits:
    mark = "HIT" if c["id"] in q["gold_ids"] else "   "
    print(f"{mark} {score:.3f} [{c['id']}] {c['text'][:100]}...")

print("\nanswer:", generate_answer(q["question"], hits))
print("\nexpected:", q["answer"])