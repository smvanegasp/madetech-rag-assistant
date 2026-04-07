"""API call and LLM-as-judge evaluation for the API evaluation experiment."""

import time

import requests
from litellm import completion

from utils.models import AnswerEval
from utils.prompts import EVALUATE_ANSWER_SYSTEM_PROMPT


def call_api(
    question: str,
    api_config: dict,
) -> tuple[str, list[dict], float]:
    """
    Call the live RAG API and measure round-trip latency.

    Args:
        question: The user question to send.
        api_config: Dict with base_url, endpoint, timeout_seconds.

    Returns:
        Tuple of (answer_text, sources_list, latency_seconds).

    Raises on HTTP or connection errors (caller should catch and record).
    """
    url = f"{api_config['base_url']}{api_config['endpoint']}"
    timeout = api_config.get("timeout_seconds", 120)

    payload = {"query": question, "history": []}

    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=timeout)
    latency = time.perf_counter() - t0

    resp.raise_for_status()
    data = resp.json()

    answer = data.get("content", "")
    sources = data.get("sources", [])

    return answer, sources, latency


def judge_answer(
    question: str,
    generated_answer: str,
    reference_answer: str,
    model: str,
) -> AnswerEval:
    """
    Score a generated answer using LLM-as-a-judge with structured output.

    Returns:
        AnswerEval with accuracy, completeness, relevance scores (1-5).

    Raises on model error or invalid structured output.
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
