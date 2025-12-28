"""Tests for clustering algorithms."""

import numpy as np
import pytest

from src.clustering.kmedoids import CapacitatedKMedoids
from src.embeddings.interest_vectors import InterestVectorizer


@pytest.fixture
def sample_vectors():
    """Create sample binary interest vectors."""
    np.random.seed(42)
    return np.random.randint(0, 2, size=(30, 10))


@pytest.fixture
def sample_labels():
    """Create sample category labels."""
    return ["Category A"] * 10 + ["Category B"] * 10 + ["Category C"] * 10


def test_interest_vectorizer_fit_transform(sample_labels):
    """Test interest vectorizer."""
    vectorizer = InterestVectorizer()
    vectors = vectorizer.fit_transform(sample_labels)
    
    assert vectors.shape == (30, 3)
    assert vectorizer.n_categories == 3
    assert np.all((vectors == 0) | (vectors == 1))


def test_jaccard_similarity(sample_labels):
    """Test Jaccard similarity computation."""
    vectorizer = InterestVectorizer()
    vectors = vectorizer.fit_transform(sample_labels)
    
    similarity = vectorizer.compute_jaccard_similarity(vectors)
    
    assert similarity.shape == (30, 30)
    assert np.allclose(np.diag(similarity), 1.0)
    assert np.all((similarity >= 0) & (similarity <= 1))


def test_kmedoids_grouping(sample_vectors):
    """Test k-medoids creates valid groups."""
    kmedoids = CapacitatedKMedoids(capacity=5, random_state=42)
    groups, medoids = kmedoids.fit_predict(sample_vectors)
    
    # Check all items are assigned
    all_items = set()
    for group in groups:
        all_items.update(group)
    assert all_items == set(range(len(sample_vectors)))
    
    # Check capacity constraint
    for group in groups:
        assert len(group) <= 5
    
    # Check medoids exist
    assert len(medoids) == len(groups)


def test_kmedoids_empty_input():
    """Test k-medoids handles empty input."""
    kmedoids = CapacitatedKMedoids(capacity=5)
    groups, medoids = kmedoids.fit_predict(np.array([]).reshape(0, 10))
    
    assert groups == []
    assert medoids == []


def test_kmedoids_single_item():
    """Test k-medoids with single item."""
    vectors = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]])
    kmedoids = CapacitatedKMedoids(capacity=5)
    groups, medoids = kmedoids.fit_predict(vectors)
    
    assert len(groups) == 1
    assert groups[0] == [0]
    assert medoids[0] == 0