"""Capacitated k-medoids clustering with balanced group sizes."""

from typing import List, Optional, Tuple

import numpy as np


class CapacitatedKMedoids:
    """K-medoids clustering with capacity constraints.
    
    Ensures each cluster has at most `capacity` members, suitable
    for balanced group formation (e.g., teams of ≤5 people).
    
    Attributes:
        capacity: Maximum members per group
        max_iterations: Maximum optimization iterations
        random_state: Random seed
    """
    
    def __init__(
        self,
        capacity: int = 5,
        max_iterations: int = 10,
        random_state: int = 42
    ):
        """Initialize capacitated k-medoids.
        
        Args:
            capacity: Max group size
            max_iterations: Convergence iterations
            random_state: Random seed
        """
        self.capacity = capacity
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        
    def fit_predict(
        self,
        vectors: np.ndarray
    ) -> Tuple[List[List[int]], List[int]]:
        """Cluster vectors into balanced groups.
        
        Args:
            vectors: Binary interest vectors (n_samples, n_features)
            
        Returns:
            Tuple of (groups, medoids) where:
                - groups: List of member index lists
                - medoids: List of medoid indices
        """
        n = vectors.shape[0]
        
        if n == 0:
            return [], []
        
        # Number of groups needed
        n_groups = (n + self.capacity - 1) // self.capacity
        
        # Initialize medoids using farthest-first
        medoids = self._pick_medoids_farthest_first(vectors, n_groups)
        prev_medoids = None
        
        # Iterative refinement
        for _ in range(self.max_iterations):
            # Compute similarities to medoids
            similarities = self._jaccard_to_centers(vectors, medoids)
            
            # Assign with capacity constraints
            groups = self._assign_with_capacity(similarities, self.capacity)
            
            # Update medoids
            new_medoids = []
            for group in groups:
                if len(group) == 0:
                    # Handle empty group by picking farthest point
                    sims = self._jaccard_to_centers(vectors, medoids).max(axis=1)
                    distances = 1.0 - sims
                    candidates = np.setdiff1d(np.arange(n), medoids)
                    
                    if len(candidates) > 0:
                        pick = int(candidates[np.argmax(distances[candidates])])
                    else:
                        pick = int(np.argmax(distances))
                    
                    new_medoids.append(pick)
                else:
                    # Pick medoid within group
                    new_medoids.append(
                        self._update_medoid_in_group(vectors, np.array(group))
                    )
            
            new_medoids = np.array(new_medoids, dtype=int)
            
            # Check convergence
            if prev_medoids is not None and set(new_medoids) == set(prev_medoids):
                break
            
            prev_medoids = medoids
            medoids = new_medoids
        
        # Final assignment
        similarities = self._jaccard_to_centers(vectors, medoids)
        groups = self._assign_with_capacity(similarities, self.capacity)
        
        # Remove empty groups and update medoids
        groups = [g for g in groups if len(g) > 0]
        medoids = [
            self._update_medoid_in_group(vectors, np.array(g))
            for g in groups
        ]
        
        return groups, medoids
    
    def _pick_medoids_farthest_first(
        self,
        vectors: np.ndarray,
        k: int
    ) -> np.ndarray:
        """Initialize medoids using farthest-first traversal.
        
        Args:
            vectors: Binary vectors
            k: Number of medoids
            
        Returns:
            Array of medoid indices
        """
        n = vectors.shape[0]
        k = min(k, n)
        
        medoids = []
        
        # Random first medoid
        first = self.rng.integers(0, n)
        medoids.append(first)
        
        # Iteratively pick farthest point
        for _ in range(k - 1):
            # Compute minimum distance to existing medoids
            sims = self._jaccard_to_centers(vectors, np.array(medoids))
            max_sims = sims.max(axis=1)
            distances = 1.0 - max_sims
            
            # Pick point farthest from all medoids
            farthest = int(np.argmax(distances))
            medoids.append(farthest)
        
        return np.array(medoids, dtype=int)
    
    @staticmethod
    def _jaccard_to_centers(
        vectors: np.ndarray,
        center_indices: np.ndarray
    ) -> np.ndarray:
        """Compute Jaccard similarity to center points.
        
        Args:
            vectors: All binary vectors (n, d)
            center_indices: Indices of centers (k,)
            
        Returns:
            Similarity matrix (n, k)
        """
        centers = vectors[center_indices]  # (k, d)
        
        # Jaccard: |A ∩ B| / |A ∪ B|
        intersection = vectors @ centers.T  # (n, k)
        
        vec_sums = vectors.sum(axis=1, keepdims=True)  # (n, 1)
        center_sums = centers.sum(axis=1, keepdims=True).T  # (1, k)
        union = vec_sums + center_sums - intersection  # (n, k)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            similarity = intersection / union
            similarity[~np.isfinite(similarity)] = 0.0
        
        return similarity
    
    @staticmethod
    def _assign_with_capacity(
        similarities: np.ndarray,
        capacity: int
    ) -> List[List[int]]:
        """Greedy assignment respecting capacity constraints.
        
        Args:
            similarities: Similarity to centers (n, k)
            capacity: Max per center
            
        Returns:
            List of index lists for each center
        """
        n, k = similarities.shape
        
        # Initialize groups and capacities
        groups = [[] for _ in range(k)]
        remaining_capacity = np.full(k, capacity, dtype=int)
        
        # Sort all items by best similarity (descending)
        item_best_sims = similarities.max(axis=1)
        sorted_items = np.argsort(-item_best_sims)
        
        for item in sorted_items:
            # Find best center with remaining capacity
            item_sims = similarities[item]
            
            # Mask centers at capacity
            valid_centers = remaining_capacity > 0
            item_sims_masked = np.where(valid_centers, item_sims, -np.inf)
            
            best_center = int(np.argmax(item_sims_masked))
            
            # Assign
            groups[best_center].append(int(item))
            remaining_capacity[best_center] -= 1
        
        return groups
    
    @staticmethod
    def _update_medoid_in_group(
        vectors: np.ndarray,
        group_indices: np.ndarray
    ) -> int:
        """Find best medoid within a group.
        
        Picks the point that minimizes average distance to all
        other points in the group.
        
        Args:
            vectors: All vectors
            group_indices: Indices of group members
            
        Returns:
            Index of best medoid
        """
        if len(group_indices) == 0:
            raise ValueError("Cannot update medoid of empty group")
        
        if len(group_indices) == 1:
            return int(group_indices[0])
        
        # Get group vectors
        group_vecs = vectors[group_indices]
        
        # Compute pairwise Jaccard
        intersection = group_vecs @ group_vecs.T
        row_sums = group_vecs.sum(axis=1, keepdims=True)
        union = row_sums + row_sums.T - intersection
        
        with np.errstate(divide='ignore', invalid='ignore'):
            similarity = intersection / union
            similarity[~np.isfinite(similarity)] = 0.0
        
        # Average similarity for each candidate medoid
        avg_similarity = similarity.mean(axis=1)
        
        # Pick best
        best_local = int(np.argmax(avg_similarity))
        return int(group_indices[best_local])
    
    def __repr__(self) -> str:
        return (
            f"CapacitatedKMedoids(capacity={self.capacity}, "
            f"max_iter={self.max_iterations})"
        )