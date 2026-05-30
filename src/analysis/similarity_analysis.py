import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def average_within_similarity(
    embeddings: np.ndarray,
) -> float:

    sim_matrix = cosine_similarity(
        embeddings
    )

    upper = sim_matrix[
        np.triu_indices(
            sim_matrix.shape[0],
            k=1,
        )
    ]

    return float(np.mean(upper))
