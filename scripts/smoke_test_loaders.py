from pathlib import Path

from src.datasets.bitext_loader import load_bitext_prompt_examples
from src.datasets.kb_loader import load_kb_documents


def main() -> None:
    kb_path = Path("data") / "kb" / "customer_support_kb.jsonl"

    kb_documents = load_kb_documents(kb_path)
    prompt_examples = load_bitext_prompt_examples(max_samples_per_intent=2)

    print(f"Loaded KB documents: {len(kb_documents)}")
    print(f"Loaded Bitext prompt examples: {len(prompt_examples)}")

    print("\nSample KB document:")
    print(kb_documents[0])

    print("\nSample prompt example:")
    print(prompt_examples[0])


if __name__ == "__main__":
    main()
