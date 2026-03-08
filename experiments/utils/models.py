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
# Chunk model for RAG retrieval
# ---------------------------------------------------------------------------


class Result(BaseModel):
    """A result of a chunk of handbook content with metadata. This is done for
    the retrieval pipeline to return a result object that can be used by the LLM.
    The page_content is the content of the chunk, including the headline, summary, and original text.
    The metadata is the metadata of the chunk, including the id, title, and category.
    """

    page_content: str = Field(
        description="The content of the chunk, including the headline, summary, and original text",
    )
    metadata: dict = Field(
        description="The metadata of the chunk, including the id, title, and category",
    )


class Chunk(BaseModel):
    """A chunk of handbook content with optional LLM-generated headline/summary."""

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
        """Convert the chunk to a Result object."""
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
    chunks: list[Chunk]


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


class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality."""

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
