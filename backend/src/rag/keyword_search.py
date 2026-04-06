"""
BM25 keyword search over handbook documents.

Complements the semantic (embedding) search by finding exact keyword matches.
Built once at startup from HandbookDoc objects and reused for all queries.
"""

import re

from rank_bm25 import BM25Okapi

from utils.models import HandbookDoc, Result


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric, remove short tokens."""
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) > 1]


class HandbookBM25Index:
    """In-memory BM25 index over handbook documents."""

    def __init__(self, docs: list[HandbookDoc]) -> None:
        self.docs = docs
        corpus = [_tokenize(doc.content) for doc in docs]
        self.bm25 = BM25Okapi(corpus)
        print(f"[OK] BM25 index built over {len(docs)} handbook documents")

    def search(self, query: str, top_k: int = 5) -> list[Result]:
        """Return top-k handbook documents matching the query by BM25 score."""
        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        # Get top-k indices with non-zero scores
        scored = [(i, s) for i, s in enumerate(scores) if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        results = []
        for idx, _score in top:
            doc = self.docs[idx]
            # Truncate long documents to a reasonable chunk size
            content = doc.content.strip()
            if len(content) > 2000:
                content = content[:2000] + "..."
            results.append(
                Result(
                    page_content=f"Headline: {doc.title}\nSummary: Full document match via keyword search\nOriginal Text:\n{content}",
                    metadata={
                        "id": doc.id,
                        "title": doc.title,
                        "category": doc.category,
                    },
                )
            )
        return results
