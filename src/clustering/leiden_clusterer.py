"""Leiden community detection on interest similarity graphs."""

from typing import Optional, Tuple

import igraph as ig
import leidenalg
import numpy as np
from sklearn.neighbors import NearestNeighbors


class LeidenClusterer:
    """Perform Leiden clustering on interest similarity graphs.
    
    Constructs a k-NN graph from interest vectors and applies
    the Leiden community detection algorithm.
    
    Attributes:
        n_neighbors: Number of neighbors for k-NN graph
        resolution: Leiden resolution parameter
        min_similarity: Minimum edge weight threshold
        random_state: Random seed for reproducibility
    """
    
    def __init__(
        self,
        n_neighbors: int = 15,
        resolution: float = 1.0,
        min_similarity: float = 0.1,
        random_state: int = 42
    ):
        """Initialize Leiden clusterer.
        
        Args:
            n_neighbors: k for k-NN graph construction
            resolution: Higher values → more clusters
            min_similarity: Threshold for edge inclusion
            random_state: Random seed
        """
        self.n_neighbors = n_neighbors
        self.resolution = resolution
        self.min_similarity = min_similarity
        self.random_state = random_state
        self.graph_ = None
        self.labels_ = None
        
    def _build_knn_graph(
        self,
        similarity_matrix: np.ndarray
    ) -> ig.Graph:
        """Build k-NN graph from similarity matrix.
        
        Args:
            similarity_matrix: Pairwise similarity matrix
            
        Returns:
            igraph Graph object with weighted edges
        """
        n = similarity_matrix.shape[0]
        
        # For each node, find k nearest neighbors
        # Use negative similarity as distance for sklearn
        nn = NearestNeighbors(
            n_neighbors=min(self.n_neighbors + 1, n),
            metric='precomputed'
        )
        distances = 1.0 - similarity_matrix
        np.fill_diagonal(distances, 0)
        nn.fit(distances)
        
        # Get k-NN indices and similarities
        knn_distances, knn_indices = nn.kneighbors(distances)
        knn_similarities = 1.0 - knn_distances
        
        # Build edge list
        edges = []
        weights = []
        
        for i in range(n):
            for j_idx in range(1, len(knn_indices[i])):  # Skip self
                j = knn_indices[i, j_idx]
                sim = knn_similarities[i, j_idx]
                
                if sim >= self.min_similarity:
                    # Add undirected edge (only once)
                    if i < j:
                        edges.append((i, j))
                        weights.append(sim)
        
        # Create igraph
        g = ig.Graph(n=n, edges=edges, directed=False)
        g.es['weight'] = weights
        
        return g
    
    def fit(
        self,
        interest_vectors: np.ndarray,
        similarity_matrix: Optional[np.ndarray] = None
    ) -> "LeidenClusterer":
        """Fit Leiden clustering.
        
        Args:
            interest_vectors: Binary interest vectors (n_samples, n_features)
            similarity_matrix: Precomputed similarity (optional)
            
        Returns:
            Self for method chaining
        """
        # Compute similarity if not provided
        if similarity_matrix is None:
            similarity_matrix = self._compute_jaccard_similarity(interest_vectors)
        
        # Build k-NN graph
        self.graph_ = self._build_knn_graph(similarity_matrix)
        
        # Run Leiden
        partition = leidenalg.find_partition(
            self.graph_,
            leidenalg.RBConfigurationVertexPartition,
            weights='weight',
            resolution_parameter=self.resolution,
            seed=self.random_state
        )
        
        self.labels_ = np.array(partition.membership)
        
        return self
    
    def fit_predict(
        self,
        interest_vectors: np.ndarray,
        similarity_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fit and return cluster labels.
        
        Args:
            interest_vectors: Binary interest vectors
            similarity_matrix: Precomputed similarity (optional)
            
        Returns:
            Cluster labels for each sample
        """
        self.fit(interest_vectors, similarity_matrix)
        return self.labels_
    
    @staticmethod
    def _compute_jaccard_similarity(vectors: np.ndarray) -> np.ndarray:
        """Compute Jaccard similarity between binary vectors.
        
        Args:
            vectors: Binary vectors (n_samples, n_features)
            
        Returns:
            Similarity matrix (n_samples, n_samples)
        """
        intersection = vectors @ vectors.T
        row_sums = vectors.sum(axis=1, keepdims=True)
        union = row_sums + row_sums.T - intersection
        
        with np.errstate(divide='ignore', invalid='ignore'):
            similarity = intersection / union
            similarity[~np.isfinite(similarity)] = 0.0
        
        return similarity
    
    def get_cluster_sizes(self) -> np.ndarray:
        """Get size of each cluster.
        
        Returns:
            Array of cluster sizes
        """
        if self.labels_ is None:
            raise ValueError("Must fit before getting cluster sizes")
        
        unique, counts = np.unique(self.labels_, return_counts=True)
        return counts
    
    def get_cluster_stats(self) -> dict:
        """Get clustering statistics.
        
        Returns:
            Dictionary with clustering metrics
        """
        if self.labels_ is None:
            raise ValueError("Must fit before getting stats")
        
        sizes = self.get_cluster_sizes()
        
        return {
            'n_clusters': len(sizes),
            'min_size': int(sizes.min()),
            'max_size': int(sizes.max()),
            'mean_size': float(sizes.mean()),
            'median_size': float(np.median(sizes)),
            'total_nodes': len(self.labels_),
            'total_edges': self.graph_.ecount() if self.graph_ else 0
        }
    
    def __repr__(self) -> str:
        fitted = "fitted" if self.labels_ is not None else "not fitted"
        return (
            f"LeidenClusterer(n_neighbors={self.n_neighbors}, "
            f"resolution={self.resolution}, {fitted})"
        )