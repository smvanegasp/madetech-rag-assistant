import os
from pathlib import Path
from typing import List

import yaml

from .models import HandbookDoc


def load_handbook_documents(handbook_dir: Path) -> List[HandbookDoc]:
    """
    Load all handbook documents from the data/handbook directory recursively.

    This function is called once at application startup to load all handbook
    content into memory. It recursively scans for .md files and parses them
    into HandbookDoc objects.

    Metadata extraction strategy:
    1. Try to parse YAML frontmatter (id, title, category)
    2. If missing, generate from file path
    3. If still missing, extract from file content (e.g., first H1 heading)

    Example frontmatter:
    ```yaml
    ---
    id: vacation-policy
    title: Vacation Policy
    category: Benefits
    ---
    ```

    Folder structure determines category:
    - data/handbook/benefits/vacation.md → category: "Benefits"
    - data/handbook/it-security/vpn.md → category: "It Security"

    Returns:
        List[HandbookDoc]: Parsed handbook documents with metadata

    Raises:
        FileNotFoundError: If data/handbook directory doesn't exist
    """
    documents = []

    if not handbook_dir.exists():
        raise FileNotFoundError(f"Handbook directory not found: {handbook_dir}")

    # Recursively find all markdown files
    md_files = sorted(handbook_dir.rglob("*.md"))

    for md_file in md_files:
        try:
            # Read file with BOM handling (some editors add UTF-8 BOM)
            with open(md_file, "r", encoding="utf-8-sig") as f:
                content = f.read()

            # Calculate relative path from handbook directory
            # Example: benefits/vacation_policy.md
            relative_path = md_file.relative_to(handbook_dir)

            # Extract category from folder structure
            # Example: benefits/vacation_policy.md → "Benefits"
            category = (
                relative_path.parts[0] if len(relative_path.parts) > 1 else "General"
            )
            # Convert to title case: "it-security" → "It Security"
            category = category.replace("_", " ").replace("-", " ").title()

            # Generate document ID from path
            # Example: benefits/vacation_policy.md → "benefits-vacation_policy"
            doc_id = str(relative_path.with_suffix("")).replace(os.sep, "-")

            # Default title from filename
            # Example: vacation_policy.md → "Vacation Policy"
            title = md_file.stem.replace("_", " ").replace("-", " ").title()

            markdown_content = content

            # Check for YAML frontmatter and process it
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    # IMPORTANT: Always strip frontmatter from content first
                    # This ensures clean markdown even if YAML parsing fails
                    markdown_content = parts[2].strip()

                    # Try to parse YAML for metadata
                    try:
                        frontmatter = yaml.safe_load(parts[1])
                        # Override default values with frontmatter if present
                        if frontmatter:
                            doc_id = frontmatter.get("id", doc_id)
                            title = frontmatter.get("title", title)
                            category = frontmatter.get("category", category)
                    except yaml.YAMLError as e:
                        print(
                            f"Warning: Invalid YAML frontmatter in {relative_path}: {e}"
                        )
                        # Content is already stripped above, so we continue
                        # with default metadata

            # Try to extract title from first H1 heading if we're still using
            # the filename-derived title
            if title == md_file.stem.replace("_", " ").replace("-", " ").title():
                lines = markdown_content.split("\n")
                for line in lines:
                    if line.startswith("# "):
                        title = line[2:].strip()
                        break

            # Create HandbookDoc object
            doc = HandbookDoc(
                id=doc_id, title=title, category=category, content=markdown_content
            )
            documents.append(doc)

        except Exception as e:
            # Fail gracefully: log error and continue with other files
            print(f"Error loading {md_file.relative_to(handbook_dir)}: {e}")
            continue

    return documents
