from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_kb_documents(kb_path: str | Path) -> list[dict[str, Any]]:
    """Load handcrafted RAG knowledge base documents from a JSONL file."""
    path = Path(kb_path)

    if not path.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {path}")

    documents: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                document = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}") from exc

            required_fields = {"doc_id", "title", "text"}
            missing_fields = required_fields - set(document)

            if missing_fields:
                raise ValueError(
                    f"Missing fields {missing_fields} on line {line_number} in {path}"
                )

            documents.append(document)

    return documents
