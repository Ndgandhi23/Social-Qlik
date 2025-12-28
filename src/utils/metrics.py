"""Clustering evaluation metrics."""

from typing import Dict

import numpy as np
from sklearn.metrics import silhouette_score


def compute_intra_cluster_similarity(
    vectors: np.ndarray,
    labels: np.ndarray,
    metric: str = 'jaccard'
) -> float:
    """Compute average within-cluster similarity.
    
    Args:
        vectors: Binary or continuous vectors
        labels: Cluster assignments
        metric: 'jaccard' or 'cosine'
        
    Returns:
        Average intra-cluster similarity
    """
    similarities = []
    
    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        cluster_vectors = vectors[cluster_mask]
        
        if len(cluster_vectors) < 2:
            continue
        
        if metric == 'jaccard':
            sim = _jaccard_similarity_matrix(cluster_vectors)
        else:
            sim = _cosine_similarity_matrix(cluster_vectors)
        
        # Get upper triangle (avoid diagonal and duplicates)
        triu_indices = np.triu_indices_from(sim, k=1)
        cluster_sims = sim[triu_indices]
        
        if len(cluster_sims) > 0:
            similarities.append(cluster_sims.mean())
    
    return float(np.mean(similarities)) if similarities else 0.0


def compute_inter_cluster_distance(
    vectors: np.ndarray,
    labels: np.ndarray,
    metric: str = 'jaccard'
) -> float:
    """Compute average between-cluster distance.
    
    Args:
        vectors: Binary or continuous vectors
        labels: Cluster assignments
        metric: 'jaccard' or 'cosine'
        
    Returns:
        Average inter-cluster distance (higher is better)
    """
    cluster_centroids = []
    cluster_ids = np.unique(labels)
    
    # Compute cluster centroids
    for cluster_id in cluster_ids:
        cluster_mask = labels == cluster_id
        centroid = vectors[cluster_mask].mean(axis=0)
        cluster_centroids.append(centroid)
    
    centroids = np.array(cluster_centroids)
    
    # Compute pairwise distances between centroids
    if metric == 'jaccard':
        sim = _jaccard_similarity_matrix(centroids)
        distance = 1.0 - sim
    else:
        sim = _cosine_similarity_matrix(centroids)
        distance = 1.0 - sim
    
    # Get upper triangle
    triu_indices = np.triu_indices_from(distance, k=1)
    distances = distance[triu_indices]
    
    return float(distances.mean()) if len(distances) > 0 else 0.0


def compute_silhouette(
    vectors: np.ndarray,
    labels: np.ndarray,
    metric: str = 'jaccard'
) -> float:
    """Compute silhouette coefficient.
    
    Args:
        vectors: Binary or continuous vectors
        labels: Cluster assignments
        metric: 'jaccard' or 'cosine'
        
    Returns:
        Silhouette score (-1 to 1, higher is better)
    """
    if len(np.unique(labels)) < 2:
        return 0.0
    
    # Convert to distance matrix for sklearn
    if metric == 'jaccard':
        sim = _jaccard_similarity_matrix(vectors)
    else:
        sim = _cosine_similarity_matrix(vectors)
    
    distance = 1.0 - sim
    
    try:
        score = silhouette_score(distance, labels, metric='precomputed')
        return float(score)
    except:
        return 0.0


def compute_group_balance(group_assignments: np.ndarray) -> Dict[str, float]:
    """Compute group size balance metrics.
    
    Args:
        group_assignments: Group ID for each item
        
    Returns:
        Dictionary with balance metrics
    """
    unique, counts = np.unique(group_assignments, return_counts=True)
    
    return {
        'n_groups': len(unique),
        'min_size': int(counts.min()),
        'max_size': int(counts.max()),
        'mean_size': float(counts.mean()),
        'std_size': float(counts.std()),
        'balance_ratio': float(counts.min() / counts.max()) if counts.max() > 0 else 0.0
    }


def evaluate_clustering(
    interest_vectors: np.ndarray,
    group_assignments: np.ndarray,
    leiden_labels: np.ndarray
) -> Dict[str, float]:
    """Comprehensive clustering evaluation.
    
    Args:
        interest_vectors: Binary interest vectors
        group_assignments: Final group assignments
        leiden_labels: Leiden cluster labels
        
    Returns:
        Dictionary with all evaluation metrics
    """
    metrics = {}
    
    # Intra-cluster similarity
    metrics['avg_intra_similarity'] = compute_intra_cluster_similarity(
        interest_vectors, group_assignments, metric='jaccard'
    )
    
    # Inter-cluster distance
    metrics['avg_inter_distance'] = compute_inter_cluster_distance(
        interest_vectors, group_assignments, metric='jaccard'
    )
    
    # Silhouette score
    metrics['silhouette_groups'] = compute_silhouette(
        interest_vectors, group_assignments, metric='jaccard'
    )
    metrics['silhouette_leiden'] = compute_silhouette(
        interest_vectors, leiden_labels, metric='jaccard'
    )
    
    # Group balance
    balance = compute_group_balance(group_assignments)
    metrics.update({f'balance_{k}': v for k, v in balance.items()})
    
    return metrics


def _jaccard_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise Jaccard similarity."""
    intersection = vectors @ vectors.T
    row_sums = vectors.sum(axis=1, keepdims=True)
    union = row_sums + row_sums.T - intersection
    
    with np.errstate(divide='ignore', invalid='ignore'):
        similarity = intersection / union
        similarity[~np.isfinite(similarity)] = 0.0
    
    return similarity


def _cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # Avoid division by zero
    normalized = vectors / norms
    return normalized @ normalized.T