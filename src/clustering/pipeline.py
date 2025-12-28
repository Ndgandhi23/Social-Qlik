"""End-to-end clustering pipeline."""

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.clustering.kmedoids import CapacitatedKMedoids
from src.clustering.leiden_clusterer import LeidenClusterer
from src.embeddings.interest_vectors import InterestVectorizer
from src.embeddings.mpnet_embedder import MPNetEmbedder
from src.visualization.umap_plotter import UMAPPlotter


class ClusteringPipeline:
    """End-to-end pipeline for interest-based clustering.
    
    Orchestrates: embedding → interest vectors → Leiden → k-medoids.
    
    Attributes:
        embedder: MPNet semantic embedder
        vectorizer: Interest vector builder
        leiden: Leiden community detector
        kmedoids: Capacitated k-medoids clusterer
        plotter: UMAP visualization tool
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        leiden_resolution: float = 1.0,
        leiden_n_neighbors: int = 15,
        group_capacity: int = 5,
        kmedoids_max_iter: int = 10,
        random_state: int = 42
    ):
        """Initialize the pipeline.
        
        Args:
            embedding_model: HuggingFace model name
            leiden_resolution: Leiden clustering resolution
            leiden_n_neighbors: k for k-NN graph
            group_capacity: Max members per group
            kmedoids_max_iter: K-medoids iterations
            random_state: Random seed
        """
        self.embedder = MPNetEmbedder(model_name=embedding_model)
        self.vectorizer = InterestVectorizer()
        self.leiden = LeidenClusterer(
            n_neighbors=leiden_n_neighbors,
            resolution=leiden_resolution,
            random_state=random_state
        )
        self.kmedoids = CapacitatedKMedoids(
            capacity=group_capacity,
            max_iterations=kmedoids_max_iter,
            random_state=random_state
        )
        self.plotter = UMAPPlotter(random_state=random_state)
        
        # Store results
        self.texts_ = None
        self.labels_ = None
        self.embeddings_ = None
        self.interest_vectors_ = None
        self.leiden_labels_ = None
        self.groups_by_cluster_ = None
        self.medoids_by_cluster_ = None
        self.group_global_ = None
        
    def fit(
        self,
        texts: List[str],
        labels: List[str]
    ) -> Dict:
        """Run the complete clustering pipeline.
        
        Args:
            texts: Text descriptions
            labels: Category labels
            
        Returns:
            Dictionary with clustering results and statistics
        """
        print("🚀 Starting clustering pipeline...")
        
        # Store inputs
        self.texts_ = texts
        self.labels_ = labels
        n = len(texts)
        
        # Step 1: Generate embeddings
        print(f"\n📊 Generating embeddings for {n} texts...")
        self.embeddings_ = self.embedder.encode(texts)
        print(f"   ✓ Embeddings shape: {self.embeddings_.shape}")
        
        # Step 2: Build interest vectors
        print("\n🎯 Building interest vectors...")
        self.interest_vectors_ = self.vectorizer.fit_transform(labels)
        print(f"   ✓ Interest vectors shape: {self.interest_vectors_.shape}")
        print(f"   ✓ Categories: {self.vectorizer.n_categories}")
        
        # Step 3: Leiden clustering
        print("\n🕸️  Running Leiden clustering...")
        self.leiden_labels_ = self.leiden.fit_predict(self.interest_vectors_)
        leiden_stats = self.leiden.get_cluster_stats()
        print(f"   ✓ Leiden clusters: {leiden_stats['n_clusters']}")
        print(f"   ✓ Cluster sizes: {leiden_stats['min_size']} - {leiden_stats['max_size']}")
        
        # Step 4: K-medoids within each Leiden cluster
        print("\n👥 Forming balanced groups with k-medoids...")
        self.groups_by_cluster_ = {}
        self.medoids_by_cluster_ = {}
        
        for cluster_id in np.unique(self.leiden_labels_):
            # Get indices for this cluster
            cluster_mask = self.leiden_labels_ == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            # Extract interest vectors for this cluster
            cluster_vectors = self.interest_vectors_[cluster_indices]
            
            # Run k-medoids
            local_groups, local_medoids = self.kmedoids.fit_predict(cluster_vectors)
            
            # Map back to global indices
            global_groups = [
                [int(cluster_indices[i]) for i in group]
                for group in local_groups
            ]
            global_medoids = [int(cluster_indices[i]) for i in local_medoids]
            
            self.groups_by_cluster_[int(cluster_id)] = global_groups
            self.medoids_by_cluster_[int(cluster_id)] = global_medoids
        
        # Create global group assignments
        self._assign_global_groups()
        
        # Summary
        total_groups = sum(len(gs) for gs in self.groups_by_cluster_.values())
        print(f"   ✓ Total groups formed: {total_groups}")
        
        stats = self._compute_statistics()
        print("\n✨ Pipeline complete!")
        
        return stats
    
    def _assign_global_groups(self):
        """Assign global group IDs to all items."""
        n = len(self.texts_)
        self.group_global_ = np.full(n, -1, dtype=int)
        
        group_id = 0
        for cluster_id in sorted(self.groups_by_cluster_.keys()):
            for group in self.groups_by_cluster_[cluster_id]:
                for idx in group:
                    self.group_global_[idx] = group_id
                group_id += 1
    
    def _compute_statistics(self) -> Dict:
        """Compute clustering statistics."""
        group_sizes = []
        for groups in self.groups_by_cluster_.values():
            group_sizes.extend([len(g) for g in groups])
        
        return {
            'n_items': len(self.texts_),
            'n_categories': self.vectorizer.n_categories,
            'n_leiden_clusters': len(np.unique(self.leiden_labels_)),
            'n_groups': len(group_sizes),
            'group_size_min': int(np.min(group_sizes)),
            'group_size_max': int(np.max(group_sizes)),
            'group_size_mean': float(np.mean(group_sizes)),
            'leiden_stats': self.leiden.get_cluster_stats()
        }
    
    def get_group_members(self, group_id: int) -> List[Tuple[int, str, str]]:
        """Get members of a specific group.
        
        Args:
            group_id: Global group ID
            
        Returns:
            List of (index, text, label) tuples
        """
        indices = np.where(self.group_global_ == group_id)[0]
        return [
            (int(idx), self.texts_[idx], self.labels_[idx])
            for idx in indices
        ]
    
    def visualize_clusters(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """Visualize Leiden clusters in UMAP space.
        
        Args:
            save_path: Path to save figure
            figsize: Figure dimensions
        """
        self.plotter.plot_leiden_clusters(
            self.interest_vectors_,
            self.leiden_labels_,
            save_path=save_path,
            figsize=figsize
        )
    
    def visualize_groups(
        self,
        cluster_id: int,
        highlight_medoids: bool = True,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """Visualize groups within a Leiden cluster.
        
        Args:
            cluster_id: Leiden cluster to visualize
            highlight_medoids: Show medoid markers
            save_path: Path to save figure
            figsize: Figure dimensions
        """
        self.plotter.plot_cluster_groups(
            self.interest_vectors_,
            self.leiden_labels_,
            cluster_id,
            self.groups_by_cluster_,
            self.medoids_by_cluster_,
            highlight_medoids=highlight_medoids,
            save_path=save_path,
            figsize=figsize
        )
    
    def export_results(self) -> Dict:
        """Export all clustering results.
        
        Returns:
            Dictionary with all results
        """
        return {
            'texts': self.texts_,
            'labels': self.labels_,
            'embeddings': self.embeddings_,
            'interest_vectors': self.interest_vectors_,
            'leiden_labels': self.leiden_labels_,
            'group_assignments': self.group_global_,
            'groups_by_cluster': self.groups_by_cluster_,
            'medoids_by_cluster': self.medoids_by_cluster_
        }
    
    def __repr__(self) -> str:
        if self.texts_ is None:
            return "ClusteringPipeline(not fitted)"
        
        n_clusters = len(np.unique(self.leiden_labels_))
        n_groups = len(set(self.group_global_))
        
        return (
            f"ClusteringPipeline(n_items={len(self.texts_)}, "
            f"n_clusters={n_clusters}, n_groups={n_groups})"
        )