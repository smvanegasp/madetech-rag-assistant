"""LLM-as-judge evaluation of RAG answer quality."""

from tenacity import retry, wait_exponential

from litellm import completion
from openai import OpenAI
from rag import answer_question

from utils.models import AnswerEval, QAPairWithTS
from utils.prompts import EVALUATE_ANSWER_SYSTEM_PROMPT


WAIT = wait_exponential(multiplier=1, min=10, max=240)


@retry(wait=WAIT)
def evaluate_answer(
    test: QAPairWithTS,
    config: dict,
    collection,
    openai_client: OpenAI,
    model: str,
) -> tuple[AnswerEval, str, list]:
    """
    Evaluate answer quality using LLM-as-a-judge.

    Runs RAG to generate an answer, then uses an LLM judge to score accuracy,
    completeness, and relevance against the reference answer.

    Args:
        test: QA pair with question and reference answer
        config: RAG config (use_query_rewriting, use_reranking, retrieval, etc.)
        collection: ChromaDB collection for retrieval
        openai_client: OpenAI client for embeddings
        model: LLM model for the judge (can match RAG model or differ)

    Returns:
        Tuple of (AnswerEval object, generated_answer string, retrieved_docs list)
    """
    generated_answer, retrieved_docs = answer_question(
        test.question,
        history=[],
        collection=collection,
        openai_client=openai_client,
        config=config,
    )

    user_prompt = f"""The user has provided the following:

        [QUESTION BEGINS]
        {test.question}
        [QUESTION ENDS]

        [GENERATED ANSWER BEGINS]
        {generated_answer}
        [GENERATED ANSWER ENDS]

        [REFERENCE ANSWER BEGINS]
        {test.answer}
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
    answer_eval = AnswerEval.model_validate_json(
        judge_response.choices[0].message.content
    )

    return answer_eval, generated_answer, retrieved_docs
