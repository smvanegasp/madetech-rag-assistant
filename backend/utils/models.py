"""Pydantic models for the RAG assistant experiments.

This module defines the data structures used throughout the pipeline:

- **Handbook documents**: Source content and metadata for the knowledge base
- **Chunks**: Split document segments with LLM-generated headlines/summaries for retrieval
- **QA pairs**: Question-answer pairs for evaluation dataset generation
- **Evaluation records**: Structures for assessing question quality and RAG answer quality
"""

from typing import List, Literal

from pydantic import BaseModel, Field


# =============================================================================
# Handbook Document Models
# =============================================================================


class HandbookDocMetadata(BaseModel):
    """Metadata for a handbook document, excluding the full content."""

    id: str
    title: str
    category: str


class HandbookDoc(HandbookDocMetadata):
    """A handbook document including its full content and metadata."""

    content: str


# =============================================================================
# Chunk Models (RAG Retrieval)
# =============================================================================


class Result(BaseModel):
    """Retrieval result combining chunk content with document metadata.

    Produced by the retrieval pipeline for consumption by the LLM. The
    page_content bundles headline, summary, and original text into a single
    string; metadata carries id, title, and category from the source document.
    """

    page_content: str = Field(
        description="The content of the chunk, including the headline, summary, and original text",
    )
    metadata: dict = Field(
        description="The metadata of the chunk, including the id, title, and category",
    )


class Chunk(BaseModel):
    """A chunk of handbook content with optional LLM-generated headline and summary."""

    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query",
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions",
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way",
    )

    def as_result(self, document: HandbookDoc) -> Result:
        """Build a Result suitable for the retrieval pipeline."""
        metadata = {
            "id": document.id,
            "title": document.title,
            "category": document.category,
        }
        return Result(
            page_content=(
                f"Headline: {self.headline}\n"
                f"Summary: {self.summary}\n"
                f"Original Text:\n{self.original_text}"
            ),
            metadata=metadata,
        )


class Chunks(BaseModel):
    """A collection of Chunk objects (e.g. for structured LLM output)."""

    chunks: list[Chunk]


# =============================================================================
# QA Pair Models (Dataset Generation)
# =============================================================================


class QAPair(BaseModel):
    """A single question-answer pair derived from handbook content."""

    question: str = Field(
        description="A question that can be asked in a search engine style"
    )
    answer: str = Field(
        description="A concise factual answer to the question based on the context"
    )


class QAPairList(BaseModel):
    """A list of question-answer pairs (e.g. for structured LLM output)."""

    pairs: List[QAPair] = Field(description="A list of question-answer pairs")


# Type aliases for QA evaluation
CritiqueType = Literal["groundedness", "relevance", "standalone"]
QuestionType = Literal["single-source", "multi-source"]


class QAPairWithTS(QAPair):
    """QA pair with type (single/multi-source) and source document metadata.

    TS = type + sources. Used for evaluation datasets where we track which
    document(s) each question was derived from.
    """

    question_type: QuestionType
    doc_metadata: List[HandbookDocMetadata] = Field(
        description="Metadata of source document(s). Single-source: 1 element; multi-source: 2-3 elements."
    )


# =============================================================================
# Evaluation Models
# =============================================================================


class QuestionCritique(BaseModel):
    """A critique of a question across a single evaluation dimension."""

    rationale: str = Field(description="A rationale for the rating")
    score: int = Field(description="A score from 1 to 5")


class QuestionCritiqueWithType(QuestionCritique):
    """Question critique with the evaluation dimension (groundedness, relevance, standalone)."""

    critique_type: CritiqueType = Field(description="The type of critique")


class QAPairEvalRecord(QAPairWithTS):
    """A QA pair with full evaluation metadata for the question.

    Extends QAPairWithTS with critique scores across groundedness, relevance,
    and standalone dimensions.
    """

    critiques: List[QuestionCritiqueWithType] = Field(
        description="Critique scores across evaluation dimensions"
    )


class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of RAG answer quality.

    Scores the generated answer against the reference answer and retrieved
    context across accuracy, completeness, and relevance.
    """

    feedback: str = Field(
        description="Concise feedback on the answer quality, comparing it to the reference answer and evaluating based on the retrieved context"
    )
    accuracy: float = Field(
        description="How factually correct is the answer compared to the reference answer? 1 (wrong. any wrong answer must score 1) to 5 (ideal - perfectly accurate). An acceptable answer would score 3."
    )
    completeness: float = Field(
        description="How complete is the answer in addressing all aspects of the question? 1 (very poor - missing key information) to 5 (ideal - all the information from the reference answer is provided completely). Only answer 5 if ALL information from the reference answer is included."
    )
    relevance: float = Field(
        description="How relevant is the answer to the specific question asked? 1 (very poor - off-topic) to 5 (ideal - directly addresses question and gives no additional information). Only answer 5 if the answer is completely relevant to the question and gives no additional information."
    )
