"""
This module contains ranker components that reorder documents by given criteria. Their goal is to improve overall
document retrieval results. Currently, this includes:

- EmbeddingSimilarityRanker: A ranker that ranks documents based on their similarity to a query using an external
  embedding asset.
"""

from ravelin.components.rankers.embedding_similarity_ranker import EmbeddingSimilarityRanker

__all__ = [
    "EmbeddingSimilarityRanker",
]
