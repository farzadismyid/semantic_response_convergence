from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from src.agents.baseline_agent import BaselineAgent
from src.data_loaders.bitext_loader import load_bitext_prompt_examples
from src.embeddings.sentence_transformer_engine import (
    EmbeddingEngine,
)
from src.evaluation.similarity import semantic_dispersion
from src.utils.paths import OUTPUT_DIR


def main() -> None:
    examples = load_bitext_prompt_examples(
        max_samples_per_intent=8,
    )

    grouped = defaultdict(list)

    for example in examples:
        grouped[example.intent].append(example)

    selected_intent = list(grouped.keys())[0]

    selected_examples = grouped[selected_intent]

    prompts = [e.prompt for e in selected_examples]

    print(f"\nSelected intent: {selected_intent}")
    print(f"Number of prompts: {len(prompts)}")

    agent = BaselineAgent()

    responses = []

    for i, prompt in enumerate(prompts, start=1):
        print(f"\nGenerating response {i}/{len(prompts)}")

        response = agent.generate_response(prompt)

        responses.append(response)

    embedding_engine = EmbeddingEngine()

    prompt_embeddings = embedding_engine.embed_texts(
        prompts
    )

    response_embeddings = embedding_engine.embed_texts(
        responses
    )

    prompt_dispersion = semantic_dispersion(
        prompt_embeddings
    )

    response_dispersion = semantic_dispersion(
        response_embeddings
    )

    print("\n--- RESULTS ---")
    print(f"Prompt dispersion: {prompt_dispersion:.4f}")
    print(f"Response dispersion: {response_dispersion:.4f}")

    combined = np.vstack(
        [prompt_embeddings, response_embeddings]
    )

    pca = PCA(n_components=2)

    reduced = pca.fit_transform(combined)

    prompt_points = reduced[: len(prompts)]
    response_points = reduced[len(prompts):]

    plt.figure(figsize=(10, 8))

    plt.scatter(
        prompt_points[:, 0],
        prompt_points[:, 1],
        label="Prompts",
        alpha=0.8,
    )

    plt.scatter(
        response_points[:, 0],
        response_points[:, 1],
        label="Responses",
        alpha=0.8,
    )

    for i in range(len(prompts)):
        plt.plot(
            [
                prompt_points[i, 0],
                response_points[i, 0],
            ],
            [
                prompt_points[i, 1],
                response_points[i, 1],
            ],
            alpha=0.3,
        )

    plt.legend()

    plt.title(
        f"Semantic Convergence: {selected_intent}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "baseline_convergence_plot.png"
    )

    plt.savefig(output_path, dpi=300)

    print(f"\nSaved plot to:\n{output_path}")


if __name__ == "__main__":
    main()
