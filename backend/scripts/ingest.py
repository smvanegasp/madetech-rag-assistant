import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

MODEL = "gpt-4.1-nano"

# Paths relative to backend/scripts/
DB_NAME = str(Path(__file__).parent.parent / "data" / "vector_db")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "data" / "handbook")

load_dotenv(override=True)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


def fetch_documents():
    """
    Recursively load all markdown files from the handbook directory.
    Extracts metadata including category (from folder path), title, and source file.
    """
    documents = []
    handbook_path = Path(KNOWLEDGE_BASE)

    if not handbook_path.exists():
        raise FileNotFoundError(f"Handbook directory not found: {KNOWLEDGE_BASE}")

    # Find all markdown files recursively
    md_files = list(handbook_path.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files")

    for md_file in md_files:
        try:
            # Read file content (utf-8-sig automatically strips BOM if present)
            with open(md_file, "r", encoding="utf-8-sig") as f:
                content = f.read()
            
            # Extract category from the folder structure
            relative_path = md_file.relative_to(handbook_path)
            category = (
                relative_path.parts[0] if len(relative_path.parts) > 1 else "general"
            )

            # Generate document ID from path
            doc_id = str(relative_path.with_suffix("")).replace(os.sep, "-")

            # Extract title from filename
            title = md_file.stem.replace("_", " ").replace("-", " ").title()

            # Strip YAML frontmatter if present
            markdown_content = content
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    # ALWAYS strip the frontmatter from content first
                    markdown_content = parts[2].strip()
                    
                    # Then try to parse YAML for metadata
                    try:
                        frontmatter = yaml.safe_load(parts[1])
                        # Override with frontmatter values if present
                        if frontmatter:
                            doc_id = frontmatter.get("id", doc_id)
                            title = frontmatter.get("title", title)
                            category = frontmatter.get("category", category)
                    except yaml.YAMLError as e:
                        print(f"Warning: Invalid YAML in {relative_path}: {e}")

            # Create document with clean content (no frontmatter)
            doc = Document(
                page_content=markdown_content,
                metadata={
                    "doc_id": doc_id,
                    "doc_type": category,
                    "category": category,
                    "title": title,
                    "source_file": str(relative_path),
                    "source": str(md_file),
                }
            )
            documents.append(doc)

        except Exception as e:
            print(f"Error loading {md_file}: {e}")
            continue

    print(f"Successfully loaded {len(documents)} documents")
    return documents


def create_chunks(documents):
    """
    Split documents into chunks while preserving metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)

    # Add chunk index to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


def create_embeddings(chunks):
    """
    Create vector embeddings and store in Chroma database.
    Deletes existing collection if present.
    """
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(DB_NAME), exist_ok=True)

    # Delete existing collection if present
    if os.path.exists(DB_NAME):
        try:
            Chroma(
                persist_directory=DB_NAME, embedding_function=embeddings
            ).delete_collection()
            print("Deleted existing vector database")
        except Exception as e:
            print(f"Note: Could not delete existing collection: {e}")

    # Create new vector store
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_NAME
    )

    # Display statistics
    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(
        f"✓ Vector database created: {count:,} vectors with {dimensions:,} dimensions"
    )
    print(f"✓ Database location: {DB_NAME}")
    return vectorstore


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")
