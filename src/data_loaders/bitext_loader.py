from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import load_dataset


@dataclass(frozen=True)
class PromptExample:
    prompt_id: str
    intent: str
    prompt: str
    response: str | None
    metadata: dict[str, Any]


def _first_existing_column(columns: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def load_bitext_prompt_examples(
    dataset_name: str = "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    split: str = "train",
    max_samples_per_intent: int | None = None,
) -> list[PromptExample]:
    """Load Bitext as the prompt dataset.

    Bitext is used only for prompts and intent groups.
    The RAG corpus must stay separate.
    """
    dataset = load_dataset(dataset_name, split=split)
    columns = list(dataset.column_names)

    intent_col = _first_existing_column(columns, ["intent", "category", "label"])
    prompt_col = _first_existing_column(columns, ["instruction", "prompt", "text", "utterance"])
    response_col = _first_existing_column(columns, ["response", "answer", "completion"])

    if intent_col is None:
        raise ValueError(f"Could not find an intent column. Available columns: {columns}")

    if prompt_col is None:
        raise ValueError(f"Could not find a prompt column. Available columns: {columns}")

    examples: list[PromptExample] = []
    intent_counts: dict[str, int] = {}

    for index, row in enumerate(dataset):
        intent = str(row[intent_col])

        if max_samples_per_intent is not None:
            current_count = intent_counts.get(intent, 0)

            if current_count >= max_samples_per_intent:
                continue

            intent_counts[intent] = current_count + 1

        prompt = str(row[prompt_col])
        response = str(row[response_col]) if response_col is not None else None

        metadata = {
            "source_dataset": dataset_name,
            "split": split,
            "raw_index": index,
            "available_columns": columns,
        }

        examples.append(
            PromptExample(
                prompt_id=f"bitext_{index}",
                intent=intent,
                prompt=prompt,
                response=response,
                metadata=metadata,
            )
        )

    return examples
