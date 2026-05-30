from collections import Counter
import pandas as pd


def count_missing_prompts(
    prompts: list[str],
) -> int:
    return sum(
        1
        for prompt in prompts
        if prompt is None or str(prompt).strip() == ""
    )


def count_duplicates(
    prompts: list[str],
) -> int:
    counts = Counter(prompts)

    return sum(
        count - 1
        for count in counts.values()
        if count > 1
    )


def prompt_lengths(
    prompts: list[str],
) -> list[int]:
    return [
        len(prompt.split())
        for prompt in prompts
    ]
