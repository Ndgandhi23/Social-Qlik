"""Tests for embedding generation."""

import numpy as np
import pytest

from src.embeddings.mpnet_embedder import MPNetEmbedder


@pytest.fixture
def sample_texts():
    return [
        "join the weekend volleyball team practice",
        "calculus homework help study session",
        "photography sunset shoot meetup"
    ]


@pytest.fixture
def embedder():
    return MPNetEmbedder(device='cpu', show_progress=False)


def test_embedder_initialization(embedder):
    """Test embedder initializes correctly."""
    assert embedder.device == 'cpu'
    assert embedder.batch_size == 32
    assert embedder.embedding_dim == 384


def test_encode_shape(embedder, sample_texts):
    """Test encoding produces correct shape."""
    embeddings = embedder.encode(sample_texts)
    
    assert embeddings.shape == (len(sample_texts), 384)
    assert isinstance(embeddings, np.ndarray)


def test_encode_normalization(embedder, sample_texts):
    """Test embeddings are normalized."""
    embeddings = embedder.encode(sample_texts, normalize=True)
    
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_array_almost_equal(norms, np.ones(len(sample_texts)), decimal=5)


def test_encode_batches(embedder):
    """Test batch encoding works."""
    texts = [f"sample text {i}" for i in range(100)]
    embeddings = embedder.encode_batches(texts, batch_size=10)
    
    assert embeddings.shape == (100, 384)


def test_empty_input(embedder):
    """Test handling of empty input."""
    embeddings = embedder.encode([])
    assert embeddings.shape == (0, 384)