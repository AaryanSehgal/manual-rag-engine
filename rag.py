import os
import math
import json
from dotenv import load_dotenv
from openai import OpenAI
from cache import cache, cache_key, save_cache

load_dotenv()
client = OpenAI()


def get_embedding(text):
    key = cache_key(text)
    if key in cache:
        return cache[key]

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    vector = response.data[0].embedding
    cache[key] = vector
    return vector


def dot_product(vector_a, vector_b):
    total = 0
    for num_a, num_b in zip(vector_a, vector_b):
        total += num_a * num_b
    return total


def magnitude(vector):
    sum_of_squares = 0
    for num in vector:
        sum_of_squares += num * num
    return math.sqrt(sum_of_squares)


def cosine_similarity(vector_a, vector_b):
    numerator = dot_product(vector_a, vector_b)
    denominator = magnitude(vector_a) * magnitude(vector_b)
    return numerator / denominator


def load_corpus():
    with open("corpus.json", "r") as f:
        corpus = json.load(f)
    for c in corpus:
        c["embedding"] = get_embedding(c["text"])
    save_cache()
    return corpus


def retrieve(question, corpus, k=5, filters=None):
    q_emb = get_embedding(question)

    candidates = corpus
    if filters:
        candidates = [c for c in corpus if all(c.get(key) == val for key, val in filters.items())]

    scored = []
    for c in candidates:
        scored.append((cosine_similarity(q_emb, c["embedding"]), c))

    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:k]


def generate_answer(question, chunks):
    context = "\n\n".join([c["text"] for score, c in chunks])

    system_prompt = "answer the question using only the context below, dont use anything you already know. if the context doesnt have the answer just say you dont know"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"context: {context}\n\nquestion: {question}"}
        ]
    )

    return response.choices[0].message.content