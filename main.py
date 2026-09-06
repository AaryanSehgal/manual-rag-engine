from fastapi import FastAPI
from pydantic import BaseModel
from rag import get_embedding, retriver, generate_answer

app = FastAPI()

# load and chunk the document ONCE, when the server starts
with open("document.txt", "r") as file:
    text = file.read()

chunks = text.split("\n\n")
chunk_embeddings = [get_embedding(c) for c in chunks]


class Question(BaseModel):
    query: str


@app.post("/ask")
def ask(q: Question):
    best_chunk, best_score = retriver(chunks, chunk_embeddings, q.query)
    answer = generate_answer(q.query, best_chunk)
    return {
        "answer": answer,
        "matched_chunk": best_chunk,
        "score": round(best_score, 4)
    }