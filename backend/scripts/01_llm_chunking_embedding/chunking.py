"""LLM-based semantic chunking of handbook documents."""

from functools import partial
from multiprocessing import Pool

from litellm import completion
from tenacity import retry, wait_exponential
from tqdm import tqdm

from utils.models import Chunks, HandbookDoc, Result
from utils.prompts import CHUNK_GENERATION_SYSTEM_PROMPT


@retry(wait=wait_exponential(multiplier=1, min=10, max=240))
def generate_chunks_from_document(
    document: HandbookDoc,
    model: str,
    max_tokens: int,
    average_chunk_size: int,
) -> list[Result]:
    """
    Split a handbook document into semantic chunks using an LLM.

    Each chunk includes a headline, summary, and original text. Uses
    structured output (Pydantic) for reliable parsing. Retries on
    transient API failures with exponential backoff.

    Args:
        document: The handbook document to chunk.
        model: LLM model identifier (e.g. groq/openai/gpt-oss-20b).
        max_tokens: Maximum tokens for the LLM response.
        average_chunk_size: Target size in characters to estimate
            number of chunks; the LLM may produce more or fewer.

    Returns:
        List of Result objects (page_content + metadata) ready for
        embedding and storage.
    """
    number_chunks = (len(document.content) // average_chunk_size) + 1
    user_prompt = (
        f"The user has provided the following document from Made Tech's handbook:\n\n"
        f"[DOCUMENT METADATA]\n"
        f"id: {document.id}\n"
        f"title: {document.title}\n"
        f"category: {document.category}\n"
        f"[DOCUMENT METADATA END]\n\n"
        f"[DOCUMENTS BEGIN]\n\n{document.content}\n\n[DOCUMENTS END]\n\n\n"
        f"This document should probably be split into at least {number_chunks} chunks, "
        f"but you can have more or less as appropriate, ensuring that there are "
        f"individual chunks to answer specific questions"
    )
    messages = [
        {"role": "system", "content": CHUNK_GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    response = completion(
        model=model,
        messages=messages,
        response_format=Chunks,
        max_tokens=max_tokens,
    )
    reply = response.choices[0].message.content
    chunks = Chunks.model_validate_json(reply).chunks
    return [chunk.as_result(document) for chunk in chunks]


def sequential_chunk_generation(
    documents: list[HandbookDoc],
    model: str,
    max_tokens: int,
    average_chunk_size: int,
) -> list[Result]:
    """
    Generate chunks for all documents sequentially with a progress bar.

    Processes documents one by one to avoid rate limits and provide
    clear progress feedback.

    Args:
        documents: List of handbook documents to chunk.
        model: LLM model identifier.
        max_tokens: Maximum tokens per LLM response.
        average_chunk_size: Target chunk size for chunk count estimation.

    Returns:
        Combined list of all Result chunks from every document.
    """
    all_chunks: list[Result] = []
    for doc in tqdm(documents, desc="Generating chunks"):
        all_chunks += generate_chunks_from_document(
            doc,
            model=model,
            max_tokens=max_tokens,
            average_chunk_size=average_chunk_size,
        )
    return all_chunks


def parallel_chunk_generation(
    documents: list[HandbookDoc],
    model: str,
    max_tokens: int,
    average_chunk_size: int,
    workers: int = 1,
) -> list[Result]:
    """
    Generate chunks for all documents in parallel using a process pool.

    Processes documents concurrently across workers. Use workers=1 to avoid
    rate limits; increase for faster processing when the API allows.

    Args:
        documents: List of handbook documents to chunk.
        model: LLM model identifier.
        max_tokens: Maximum tokens per LLM response.
        average_chunk_size: Target chunk size for chunk count estimation.
        workers: Number of parallel workers. Set to 1 if you hit rate limits.

    Returns:
        Combined list of all Result chunks from every document.
    """
    process_doc = partial(
        generate_chunks_from_document,
        model=model,
        max_tokens=max_tokens,
        average_chunk_size=average_chunk_size,
    )
    all_chunks: list[Result] = []
    with Pool(processes=workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_doc, documents),
            total=len(documents),
            desc="Generating chunks",
        ):
            all_chunks.extend(result)
    return all_chunks
