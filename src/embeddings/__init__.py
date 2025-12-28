"""Embedding generation modules."""

from .interest_vectors import InterestVectorizer
from .mpnet_embedder import MPNetEmbedder

__all__ = ["MPNetEmbedder", "InterestVectorizer"]