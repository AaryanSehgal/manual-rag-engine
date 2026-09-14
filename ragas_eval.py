import json
import asyncio
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecisionWithReference
from ragas.embeddings import embedding_factory
from rag import load_corpus, retrieve, generate_answer
from cache import save_cache


async def main():
    corpus = load_corpus()
    with open("evalset.json") as f:
        evalset = json.load(f)

    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client, max_tokens=4000)
    embeddings = embedding_factory(provider="openai", model="text-embedding-3-small", client=client)

    faith = Faithfulness(llm=llm)
    relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    precision = ContextPrecisionWithReference(llm=llm)

    sample = evalset[:20]
    f_scores, r_scores, p_scores = [], [], []

    for i, item in enumerate(sample):
        print(f"{i+1}/{len(sample)}", end="\r")

        retrieved = retrieve(item["question"], corpus, k=3)
        contexts = [c["text"] for score, c in retrieved]
        answer = generate_answer(item["question"], retrieved)

        f = await faith.ascore(
            user_input=item["question"],
            response=answer,
            retrieved_contexts=contexts
        )
        r = await relevancy.ascore(
            user_input=item["question"],
            response=answer
        )
        p = await precision.ascore(
            user_input=item["question"],
            retrieved_contexts=contexts,
            reference=item["answer"]
        )

        f_scores.append(f.value)
        r_scores.append(r.value)
        p_scores.append(p.value)

    save_cache()
    print()
    print(f"faithfulness:      {sum(f_scores)/len(f_scores):.3f}")
    print(f"answer relevancy:  {sum(r_scores)/len(r_scores):.3f}")
    print(f"context precision: {sum(p_scores)/len(p_scores):.3f}")


asyncio.run(main())