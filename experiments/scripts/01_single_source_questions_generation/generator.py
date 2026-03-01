"""QA generation and critique logic for single-source eval questions."""

import logging
import random
from typing import List

from litellm import completion
from tqdm import tqdm
from utils.models import (
    HandbookDoc,
    HandbookDocMetadata,
    QAPairEvalRecord,
    QAPairList,
    QAPairWithTS,
    QuestionCritique,
    QuestionCritiqueWithType,
)
from utils.prompts import (
    GROUNDEDNESS_CRITIQUE_SYSTEM_PROMPT,
    QA_GENERATION_SYSTEM_PROMPT,
    RELEVANCE_CRITIQUE_SYSTEM_PROMPT,
    STANDALONE_CRITIQUE_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# Maps each critique dimension name to its system prompt so critique_question()
# can look up the right prompt without a long if/elif chain.
CRITIQUE_PROMPTS = {
    "groundedness": GROUNDEDNESS_CRITIQUE_SYSTEM_PROMPT,
    "relevance": RELEVANCE_CRITIQUE_SYSTEM_PROMPT,
    "standalone": STANDALONE_CRITIQUE_SYSTEM_PROMPT,
}


def generate_qa_pairs_from_single_document(
    document: HandbookDoc,
    num_questions: int = 3,
    model: str = "groq/openai/gpt-oss-20b",
) -> List[QAPairWithTS]:
    """
    Generate search-style factoid QA pairs from a single document.

    Args:
        document: The document object.
        num_questions: Number of QA pairs to generate.
        model: Model name to use for LLM.

    Returns:
        List of QAPairWithTS objects with source document metadata.
    """
    logger.debug(
        "Generating %d QA pairs from doc '%s' (id=%s) using model '%s'",
        num_questions,
        document.title,
        document.id,
        model,
    )

    # Embed the full document text and the requested count in the user turn.
    # The system prompt (QA_GENERATION_SYSTEM_PROMPT) instructs the model on
    # format and question style.
    user_prompt = (
        f"The user has provided the following document:\n\n"
        f"[DOCUMENT BEGINS]\n\n{document.content}\n\n[DOCUMENT ENDS]\n\n\n"
        f"Generate {num_questions} question-answer pairs from the document.\n\n"
        "Reply only with the question-answer pairs, nothing else."
    )
    messages = [
        {"role": "system", "content": QA_GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    response = completion(model=model, messages=messages, response_format=QAPairList)
    reply = response.choices[0].message.content

    # Parse the structured JSON response into QAPairList, then attach source
    # metadata (title, category, id) to each pair via QAPairWithTS.
    # model_dump() / model_dump(exclude=...) are Pydantic v2 serialisation helpers.
    pairs = QAPairList.model_validate_json(reply).pairs
    logger.info(
        "Doc '%s': generated %d QA pairs", document.title, len(pairs)
    )

    return [
        QAPairWithTS(
            **pair.model_dump(),
            question_type="single-source",
            doc_metadata=[
                HandbookDocMetadata(**document.model_dump(exclude={"content"}))
            ],
        )
        for pair in pairs
    ]


def critique_question(
    context: str,
    question: str,
    critique_type: str,
    model: str = "groq/openai/gpt-oss-20b",
) -> QuestionCritiqueWithType:
    """
    Critique a question against a context using a specified evaluation dimension.

    Args:
        context: The source document text.
        question: The question to evaluate.
        critique_type: One of "groundedness", "relevance", or "standalone".
        model: Model name to use for the LLM call.

    Returns:
        A QuestionCritiqueWithType with rationale, score, and critique type.
    """
    logger.debug(
        "Critiquing question on dimension '%s' using model '%s' | Q: %.80s…",
        critique_type,
        model,
        question,
    )

    system_prompt = CRITIQUE_PROMPTS[critique_type]
    # Provide both the question and its source context so the model can judge
    # whether the question is grounded in / relevant to / standalone from it.
    user_prompt = (
        f"The user has provided the following question:\n\n"
        f"[QUESTION BEGINS]\n\n{question}\n\n[QUESTION ENDS]\n\n"
        f"The user has provided the following context:\n\n"
        f"[CONTEXT BEGINS]\n\n{context}\n\n[CONTEXT ENDS]\n\n\n"
        "Reply only with your rationale for the rating and your score (1-5), nothing else."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = completion(
        model=model, messages=messages, response_format=QuestionCritique
    )
    reply = response.choices[0].message.content

    # Wrap the parsed critique with its dimension name so downstream code can
    # build a keyed score map without tracking order.
    result = QuestionCritiqueWithType(
        **QuestionCritique.model_validate_json(reply).model_dump(),
        critique_type=critique_type,
    )
    logger.debug(
        "Dimension '%s' scored %d/5", critique_type, result.score
    )
    return result


def critique_all_dimensions(
    context: str,
    question: str,
    model: str = "groq/openai/gpt-oss-20b",
) -> List[QuestionCritiqueWithType]:
    """
    Critique a question against a context across all supported evaluation dimensions.

    Args:
        context: The source document text.
        question: The question to evaluate.
        model: Model name to use for the LLM call.

    Returns:
        A list of QuestionCritiqueWithType instances, one per critique dimension.
    """
    # Run all three dimensions sequentially; each is an independent LLM call.
    critiques = [
        critique_question(context, question, crit_type, model=model)
        for crit_type in ["relevance", "standalone", "groundedness"]
    ]
    scores = {c.critique_type: c.score for c in critiques}
    logger.debug("All dimensions scored: %s", scores)
    return critiques


def generate_eval_dataset(
    documents: List[HandbookDoc],
    n_docs: int = 10,
    seed: int = 42,
    questions_per_doc: int = 3,
    model: str = "groq/openai/gpt-oss-20b",
) -> List[QAPairEvalRecord]:
    """
    Sample n_docs documents and run the full QA generation + critique pipeline.

    Args:
        documents: Full list of handbook documents to sample from.
        n_docs: Number of documents to sample.
        seed: Random seed for reproducible document sampling.
        questions_per_doc: Number of QA pairs to generate per document.
        model: Model name for QA generation and critique.

    Returns:
        List of QAPairEvalRecord instances with critiques across all dimensions.
    """
    random.seed(seed)
    # Cap at the total number of available documents to avoid a ValueError from
    # random.sample when n_docs > len(documents).
    sampled_docs = random.sample(documents, min(n_docs, len(documents)))
    logger.debug("Sampled %d documents (seed=%d) for eval generation", len(sampled_docs), seed)

    # Pre-build a content lookup so critique calls can retrieve document text
    # by id without iterating the full list each time.
    doc_content_lookup = {doc.id: doc.content for doc in sampled_docs}

    # --- Step 1: generate QA pairs for every sampled document ---
    all_qa_pairs: List[QAPairWithTS] = []
    for doc in tqdm(sampled_docs, desc="Generating QA pairs", unit="doc"):
        logger.debug("Generating QA pairs from doc '%s' (id=%s)", doc.title, doc.id)
        all_qa_pairs.extend(
            generate_qa_pairs_from_single_document(
                doc, num_questions=questions_per_doc, model=model
            )
        )
    logger.debug("Total QA pairs generated: %d", len(all_qa_pairs))

    # --- Step 2: critique every QA pair across all three dimensions ---
    eval_records = []
    for qa_pair in tqdm(all_qa_pairs, desc="Critiquing QA pairs", unit="pair"):
        logger.debug("Critiquing pair: '%s'", qa_pair.question[:80])
        # Single-source: context is the one document; multi-source would concatenate all.
        doc_ids = [m.id for m in qa_pair.doc_metadata]
        context = "\n\n---\n\n".join(
            doc_content_lookup[doc_id] for doc_id in doc_ids
        )
        eval_records.append(
            QAPairEvalRecord(
                **qa_pair.model_dump(),
                critiques=critique_all_dimensions(
                    context,
                    qa_pair.question,
                    model=model,
                ),
            )
        )

    logger.debug("Critique phase complete. %d eval records ready.", len(eval_records))
    return eval_records
