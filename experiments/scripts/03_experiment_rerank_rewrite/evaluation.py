"""LLM-as-judge evaluation of RAG answer quality with latency and failure tracking."""

import time

from litellm import completion
from openai import OpenAI
from rag import answer_question

from utils.models import AnswerEval, QAPairWithTS
from utils.prompts import EVALUATE_ANSWER_SYSTEM_PROMPT


def generate_answer(
    test: QAPairWithTS,
    config: dict,
    collection,
    openai_client: OpenAI,
) -> tuple[str, list, float]:
    """
    Run RAG to generate an answer with latency tracking.

    Returns:
        Tuple of (generated_answer, retrieved_docs, latency_seconds)

    Raises on failure (caller should catch and record).
    """
    t0 = time.perf_counter()
    generated_answer, retrieved_docs = answer_question(
        test.question,
        history=[],
        collection=collection,
        openai_client=openai_client,
        config=config,
    )
    latency = time.perf_counter() - t0
    return generated_answer, retrieved_docs, latency


def judge_answer(
    question: str,
    generated_answer: str,
    reference_answer: str,
    model: str,
) -> AnswerEval:
    """
    Score a generated answer using LLM-as-a-judge with structured output.

    Returns:
        AnswerEval with accuracy, completeness, relevance scores.

    Raises on failure (model error or invalid structured output).
    """
    user_prompt = f"""The user has provided the following:

        [QUESTION BEGINS]
        {question}
        [QUESTION ENDS]

        [GENERATED ANSWER BEGINS]
        {generated_answer}
        [GENERATED ANSWER ENDS]

        [REFERENCE ANSWER BEGINS]
        {reference_answer}
        [REFERENCE ANSWER ENDS]

        Reply with your feedback and your scores for accuracy, completeness, and relevance, nothing else."""

    judge_messages = [
        {"role": "system", "content": EVALUATE_ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    judge_response = completion(
        model=model,
        messages=judge_messages,
        response_format=AnswerEval,
    )
    return AnswerEval.model_validate_json(
        judge_response.choices[0].message.content
    )
