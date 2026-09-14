from datasets import load_dataset

passages = load_dataset("enelpol/rag-mini-bioasq", "text-corpus")["test"]
qa = load_dataset("enelpol/rag-mini-bioasq", "question-answer-passages")["test"]

print("passages:", passages)
print()
print("qa:", qa)
print()
print("sample passage:", passages[0])
print()
print("sample qa:", qa[0])
print("qa item:", qa[0])
print("type of ids:", type(qa[0]["relevant_passage_ids"]))