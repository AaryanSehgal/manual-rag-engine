import json
from rag import load_corpus, retrieve
from cache import save_cache

corpus = load_corpus()
with open("evalset.json") as f:
    evalset = json.load(f)

q = evalset[0]["question"]

print("question:", q)
print()

print("no filter (2316 passages):")
for score, c in retrieve(q, corpus, k=3):
    print(f"  {score:.3f} [{c['id']}] relevant={c['relevant']}")

print()
print("filtered to relevant=True (816 passages):")
for score, c in retrieve(q, corpus, k=3, filters={"relevant": True}):
    print(f"  {score:.3f} [{c['id']}] relevant={c['relevant']}")

print()
print("filtered to relevant=False (1500 distractors):")
for score, c in retrieve(q, corpus, k=3, filters={"relevant": False}):
    print(f"  {score:.3f} [{c['id']}] relevant={c['relevant']}")

save_cache()