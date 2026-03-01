from typing import List, Literal

from pydantic import BaseModel, Field


class HandbookDocMetadata(BaseModel):
    """Metadata for a handbook document, excluding the full content."""

    id: str
    title: str
    category: str


class HandbookDoc(HandbookDocMetadata):
    """Represents a single handbook document with metadata."""

    content: str


# ---------------------------------------------------------------------------
# Pydantic models for QA generation
# ---------------------------------------------------------------------------


class QAPair(BaseModel):
    question: str = Field(
        description="A question that can be asked in a search engine style"
    )
    answer: str = Field(
        description="A concise factual answer to the question based on the context"
    )


class QAPairList(BaseModel):
    pairs: List[QAPair] = Field(description="A list of question-answer pairs")


CritiqueType = Literal["groundedness", "relevance", "standalone"]
QuestionType = Literal["single-source", "multi-source"]


class QAPairWithTS(QAPair):
    question_type: QuestionType
    doc_metadata: List[HandbookDocMetadata] = Field(
        description="Metadata of source document(s). Single-source: 1 element; multi-source: 2-3 elements."
    )


class QuestionCritique(BaseModel):
    rationale: str = Field(description="A rationale for the rating")
    score: int = Field(description="A score from 1 to 5")


class QuestionCritiqueWithType(QuestionCritique):
    critique_type: CritiqueType = Field(description="The type of critique")


class QAPairEvalRecord(QAPairWithTS):
    critiques: List[QuestionCritiqueWithType] = Field(
        description="Critique scores across evaluation dimensions"
    )
