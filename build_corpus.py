import json
from datasets import load_dataset

passages = load_dataset("enelpol/rag-mini-bioasq", "text-corpus")["test"]
qa = load_dataset("enelpol/rag-mini-bioasq", "question-answer-passages")["test"]

qa_sample = [qa[i] for i in range(100)]

needed_ids = set()
for item in qa_sample:
    ids = item["relevant_passage_ids"]
    if isinstance(ids, str):
        ids = json.loads(ids)
    for i in ids:
        needed_ids.add(int(i))

print("questions:", len(qa_sample))
print("passages needed by those questions:", len(needed_ids))

id_to_passage = {}
for p in passages:
    if p["id"] in needed_ids:
        id_to_passage[p["id"]] = p["passage"]

print("found:", len(id_to_passage))

distractors = []
for p in passages:
    if p["id"] not in needed_ids:
        distractors.append(p)
    if len(distractors) >= 1500:
        break

corpus = []
for pid, text in id_to_passage.items():
    corpus.append({"id": pid, "text": text, "relevant": True})
for p in distractors:
    corpus.append({"id": p["id"], "text": p["passage"], "relevant": False})

print("total corpus:", len(corpus))

evalset = []
for item in qa_sample:
    ids = item["relevant_passage_ids"]
    if isinstance(ids, str):
        ids = json.loads(ids)
    evalset.append({
        "question": item["question"],
        "answer": item["answer"],
        "gold_ids": [int(i) for i in ids]
    })

with open("corpus.json", "w") as f:
    json.dump(corpus, f)
with open("evalset.json", "w") as f:
    json.dump(evalset, f)

print("saved corpus.json and evalset.json")
