"""Tests for the complete pipeline."""

import pytest
from src.clustering.pipeline import ClusteringPipeline


@pytest.fixture
def sample_data():
    """Small sample dataset for testing."""
    texts = [
        "basketball practice tonight",
        "volleyball team meeting",
        "math homework help",
        "physics study group",
        "photography workshop",
        "painting class"
    ]
    labels = [
        "Sports / Fitness",
        "Sports / Fitness",
        "Academic / Study",
        "Academic / Study",
        "Hobbies / Creative",
        "Hobbies / Creative"
    ]
    return texts, labels


def test_pipeline_initialization():
    """Test pipeline can be initialized."""
    pipeline = ClusteringPipeline(group_capacity=5)
    
    assert pipeline.kmedoids.capacity == 5
    assert pipeline.embedder.embedding_dim == 384


def test_pipeline_fit(sample_data):
    """Test pipeline can run on sample data."""
    texts, labels = sample_data
    pipeline = ClusteringPipeline(group_capacity=3)
    
    results = pipeline.fit(texts, labels)
    
    # Check results structure
    assert 'n_items' in results
    assert 'n_categories' in results
    assert 'n_groups' in results
    
    # Check values
    assert results['n_items'] == len(texts)
    assert results['n_categories'] == 3
    assert results['n_groups'] > 0


def test_pipeline_get_group_members(sample_data):
    """Test retrieving group members."""
    texts, labels = sample_data
    pipeline = ClusteringPipeline(group_capacity=3)
    pipeline.fit(texts, labels)
    
    # Get first group
    members = pipeline.get_group_members(0)
    
    # Check structure
    assert len(members) > 0
    for idx, text, label in members:
        assert isinstance(idx, int)
        assert isinstance(text, str)
        assert isinstance(label, str)


def test_pipeline_export_results(sample_data):
    """Test exporting results."""
    texts, labels = sample_data
    pipeline = ClusteringPipeline()
    pipeline.fit(texts, labels)
    
    exported = pipeline.export_results()
    
    # Check all expected keys
    assert 'texts' in exported
    assert 'labels' in exported
    assert 'embeddings' in exported
    assert 'interest_vectors' in exported
    assert 'leiden_labels' in exported
    assert 'group_assignments' in exported