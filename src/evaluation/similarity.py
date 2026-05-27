import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def average_pairwise_similarity(embeddings: np.ndarray) -> float:
    similarity_matrix = cosine_similarity(embeddings)

    n = similarity_matrix.shape[0]

    upper_triangle = similarity_matrix[
        np.triu_indices(n, k=1)
    ]

    return float(np.mean(upper_triangle))


def semantic_dispersion(embeddings: np.ndarray) -> float:
    return 1.0 - average_pairwise_similarity(embeddings)
