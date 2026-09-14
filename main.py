from fastapi import FastAPI
from pydantic import BaseModel
from rag import load_corpus, retrieve, generate_answer

app = FastAPI()

print("loading corpus...")
corpus = load_corpus()
print(f"loaded {len(corpus)} passages")


class Question(BaseModel):
    query: str
    k: int = 5
    relevant_only: bool = False


@app.get("/")
def home():
    return {"status": "running", "passages": len(corpus)}


@app.post("/ask")
def ask(q: Question):
    filters = {"relevant": True} if q.relevant_only else None
    retrieved = retrieve(q.query, corpus, k=q.k, filters=filters)
    answer = generate_answer(q.query, retrieved)

    return {
        "answer": answer,
        "sources": [
            {"id": c["id"], "score": round(score, 4), "text": c["text"][:200]}
            for score, c in retrieved
        ]
    }