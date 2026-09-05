import os
import math
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def get_embedding(text):
    """Get the embedding vector for a piece of text using OpenAI."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def dot_product(vector_a, vector_b):
    # multiply matching numbers from both vectors and add them all up
    total = 0
    for num_a, num_b in zip(vector_a, vector_b):
        total += num_a * num_b
    return total


def magnitude(vector):
    # length of the vector - square every number, add them, square root it
    sum_of_squares = 0
    for num in vector:
        sum_of_squares += num * num
    return math.sqrt(sum_of_squares)


def cosine_similarity(vector_a, vector_b):
    # cos(angle) between two vectors = dot product / (length_a * length_b)
    numerator = dot_product(vector_a, vector_b)
    denominator = magnitude(vector_a) * magnitude(vector_b)
    return numerator / denominator


with open("document.txt", "r") as file:
    text = file.read()

chunks = text.split("\n\n")

chunk_embedding = [get_embedding(c) for c in chunks]

user_input = input("what are you looking to get answer on today? ")


def retriver(chunks, chunk_embeddings, question=user_input):
    best_score = -1
    best_chunk = None
    question_embedding = get_embedding(question)

    for chunk, chunk_emb in zip(chunks, chunk_embeddings):
        score = cosine_similarity(question_embedding, chunk_emb)
        if score > best_score:
            best_score = score
            best_chunk = chunk

    print(best_chunk, best_score)
    return best_chunk


best_chunk = retriver(chunks, chunk_embedding)


# now feed the best chunk back into the model so it actually answers
# the question instead of just returning the raw text

def generate_answer(question, context):
    system_prompt = "answer the question using only the context below, dont use anything you already know. if the context doesnt have the answer just say you dont know"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"context: {context} question: {question}"}
        ]
    )

    return response.choices[0].message.content


answer = generate_answer(user_input, best_chunk)
print(answer)