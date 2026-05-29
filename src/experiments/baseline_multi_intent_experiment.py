from collections import defaultdict
import json

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from src.agents.baseline_agent import BaselineAgent
from src.datasets.bitext_loader import load_bitext_prompt_examples
from src.embeddings.sentence_transformer_engine import EmbeddingEngine
from src.evaluation.similarity import semantic_dispersion
from src.utils.experiment_paths import create_experiment_dir


def main() -> None:
    experiment_dir = create_experiment_dir("baseline_multi_intent")

    max_intents = 15
    samples_per_intent = 20

    examples = load_bitext_prompt_examples(
        max_samples_per_intent=samples_per_intent,
    )

    grouped = defaultdict(list)

    for example in examples:
        grouped[example.intent].append(example)

    selected_intents = list(grouped.keys())[:max_intents]

    agent = BaselineAgent()
    embedding_engine = EmbeddingEngine()

    all_rows = []
    all_generations = []

    for intent in tqdm(selected_intents, desc="Running intents"):
        intent_examples = grouped[intent]

        prompts = [example.prompt for example in intent_examples]
        responses = []

        for prompt in tqdm(prompts, desc=f"Generating {intent}", leave=False):
            response = agent.generate_response(prompt)
            responses.append(response)

        prompt_embeddings = embedding_engine.embed_texts(prompts)
        response_embeddings = embedding_engine.embed_texts(responses)

        prompt_dispersion = semantic_dispersion(prompt_embeddings)
        response_dispersion = semantic_dispersion(response_embeddings)

        convergence_ratio = response_dispersion / prompt_dispersion

        all_rows.append(
            {
                "condition": "A1_baseline",
                "intent": intent,
                "num_prompts": len(prompts),
                "prompt_dispersion": prompt_dispersion,
                "response_dispersion": response_dispersion,
                "convergence_ratio": convergence_ratio,
                "compression_observed": response_dispersion < prompt_dispersion,
            }
        )

        for example, response in zip(intent_examples, responses):
            all_generations.append(
                {
                    "condition": "A1_baseline",
                    "intent": intent,
                    "prompt_id": example.prompt_id,
                    "prompt": example.prompt,
                    "response": response,
                    "metadata": example.metadata,
                }
            )

    metrics_df = pd.DataFrame(all_rows)

    metrics_path = experiment_dir / "baseline_multi_intent_metrics.csv"
    generations_path = experiment_dir / "baseline_multi_intent_generations.jsonl"

    metrics_df.to_csv(metrics_path, index=False)

    with generations_path.open("w", encoding="utf-8") as file:
        for item in all_generations:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n--- BASELINE MULTI-INTENT RESULTS ---")
    print(metrics_df)

    print("\nMean prompt dispersion:")
    print(metrics_df["prompt_dispersion"].mean())

    print("\nMean response dispersion:")
    print(metrics_df["response_dispersion"].mean())

    print("\nMean convergence ratio:")
    print(metrics_df["convergence_ratio"].mean())

    print("\nCompression observed in:")
    print(metrics_df["compression_observed"].sum(), "/", len(metrics_df), "intents")

    plot_dispersion(metrics_df, experiment_dir)
    plot_convergence_ratio(metrics_df, experiment_dir)

    print(f"\nSaved metrics to:\n{metrics_path}")
    print(f"\nSaved generations to:\n{generations_path}")


def plot_dispersion(metrics_df: pd.DataFrame, experiment_dir) -> None:
    x = range(len(metrics_df))

    plt.figure(figsize=(12, 7))

    plt.plot(
        x,
        metrics_df["prompt_dispersion"],
        marker="o",
        label="Prompt dispersion",
    )

    plt.plot(
        x,
        metrics_df["response_dispersion"],
        marker="o",
        label="Response dispersion",
    )

    plt.xticks(
        x,
        metrics_df["intent"],
        rotation=45,
        ha="right",
    )

    plt.ylabel("Semantic dispersion")
    plt.title("Prompt vs Response Dispersion by Intent")
    plt.legend()
    plt.tight_layout()

    output_path = experiment_dir / "prompt_vs_response_dispersion.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_convergence_ratio(metrics_df: pd.DataFrame, experiment_dir) -> None:
    x = range(len(metrics_df))

    plt.figure(figsize=(12, 7))

    plt.bar(
        x,
        metrics_df["convergence_ratio"],
    )

    plt.axhline(
        y=1.0,
        linestyle="--",
        label="No compression threshold",
    )

    plt.xticks(
        x,
        metrics_df["intent"],
        rotation=45,
        ha="right",
    )

    plt.ylabel("Response dispersion / Prompt dispersion")
    plt.title("Baseline Convergence Ratio by Intent")
    plt.legend()
    plt.tight_layout()

    output_path = experiment_dir / "convergence_ratio_by_intent.png"
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
