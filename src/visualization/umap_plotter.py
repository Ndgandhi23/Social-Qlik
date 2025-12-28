"""UMAP-based visualization for clustering results."""

from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import umap


class UMAPPlotter:
    """Create UMAP visualizations of clustering results.
    
    Projects high-dimensional interest vectors to 2D for visualization,
    with overlays for Leiden clusters and k-medoids groups.
    
    Attributes:
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: Distance metric for UMAP
        random_state: Random seed
    """
    
    def __init__(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "jaccard",
        random_state: int = 42
    ):
        """Initialize UMAP plotter.
        
        Args:
            n_neighbors: UMAP neighborhood size
            min_dist: UMAP minimum distance
            metric: Distance metric ('jaccard', 'cosine', etc.)
            random_state: Random seed
        """
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        
    def _fit_umap(
        self,
        vectors: np.ndarray,
        n_neighbors: Optional[int] = None,
        min_dist: Optional[float] = None
    ) -> np.ndarray:
        """Fit UMAP and return 2D coordinates.
        
        Args:
            vectors: High-dimensional vectors
            n_neighbors: Override default n_neighbors
            min_dist: Override default min_dist
            
        Returns:
            2D UMAP coordinates (n_samples, 2)
        """
        reducer = umap.UMAP(
            n_neighbors=n_neighbors or self.n_neighbors,
            min_dist=min_dist or self.min_dist,
            metric=self.metric,
            random_state=self.random_state
        )
        return reducer.fit_transform(vectors)
    
    def plot_leiden_clusters(
        self,
        interest_vectors: np.ndarray,
        leiden_labels: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 300
    ):
        """Plot UMAP colored by Leiden clusters.
        
        Args:
            interest_vectors: Binary interest vectors
            leiden_labels: Leiden cluster assignments
            save_path: Path to save figure (optional)
            figsize: Figure size
            dpi: Resolution for saved figure
        """
        # Compute UMAP
        coords = self._fit_umap(interest_vectors)
        
        # Prepare colors
        cluster_ids = np.unique(leiden_labels)
        n_clusters = len(cluster_ids)
        
        cmap_name = "tab20" if n_clusters > 10 else "tab10"
        cmap = mpl.cm.get_cmap(cmap_name, n_clusters)
        
        # Map cluster IDs to indices for coloring
        cluster_to_idx = {cid: i for i, cid in enumerate(cluster_ids)}
        colors = np.array([cluster_to_idx[c] for c in leiden_labels])
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=colors,
            cmap=cmap,
            s=30,
            alpha=0.8,
            edgecolors='none'
        )
        
        # Create legend
        handles = [
            plt.Line2D(
                [0], [0],
                marker='o',
                linestyle='',
                label=f"Cluster {cid}",
                markerfacecolor=cmap(cluster_to_idx[cid]),
                markeredgecolor='none',
                markersize=8
            )
            for cid in cluster_ids
        ]
        
        ax.legend(
            handles=handles,
            title="Leiden Clusters",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.
        )
        
        ax.set_title("UMAP Projection — Leiden Clusters", fontsize=14, pad=15)
        ax.set_xlabel("UMAP-1", fontsize=12)
        ax.set_ylabel("UMAP-2", fontsize=12)
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"   💾 Saved to {save_path}")
        
        plt.show()
    
    def plot_cluster_groups(
        self,
        interest_vectors: np.ndarray,
        leiden_labels: np.ndarray,
        cluster_id: int,
        groups_by_cluster: Dict[int, List[List[int]]],
        medoids_by_cluster: Dict[int, List[int]],
        highlight_medoids: bool = True,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 300
    ):
        """Plot UMAP for a single Leiden cluster, colored by groups.
        
        Args:
            interest_vectors: Binary interest vectors
            leiden_labels: Leiden cluster assignments
            cluster_id: Which cluster to visualize
            groups_by_cluster: Mapping of cluster → groups
            medoids_by_cluster: Mapping of cluster → medoids
            highlight_medoids: Draw circles around medoids
            save_path: Path to save figure
            figsize: Figure size
            dpi: Resolution
        """
        # Get indices for this cluster
        cluster_mask = leiden_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            print(f"⚠️  No items in cluster {cluster_id}")
            return
        
        # Extract cluster vectors
        cluster_vectors = interest_vectors[cluster_indices]
        
        # Fit UMAP for this cluster (tighter parameters)
        coords = self._fit_umap(cluster_vectors, n_neighbors=10, min_dist=0.05)
        
        # Build group labels for points in this cluster
        groups = groups_by_cluster[cluster_id]
        n_groups = len(groups)
        
        group_labels = np.full(len(cluster_indices), -1, dtype=int)
        for group_idx, group in enumerate(groups):
            for global_idx in group:
                local_idx = np.where(cluster_indices == global_idx)[0][0]
                group_labels[local_idx] = group_idx
        
        # Colors
        cmap = mpl.cm.get_cmap("tab20", max(n_groups, 3))
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=group_labels,
            cmap=cmap,
            s=40,
            alpha=0.9,
            edgecolors='white',
            linewidths=0.5
        )
        
        # Highlight medoids
        if highlight_medoids and cluster_id in medoids_by_cluster:
            medoid_globals = medoids_by_cluster[cluster_id]
            medoid_locals = [
                np.where(cluster_indices == m)[0][0]
                for m in medoid_globals
            ]
            
            ax.scatter(
                coords[medoid_locals, 0],
                coords[medoid_locals, 1],
                s=150,
                facecolors='none',
                edgecolors='black',
                linewidths=2,
                label='Medoids'
            )
        
        # Legend
        handles = [
            plt.Line2D(
                [0], [0],
                marker='o',
                linestyle='',
                label=f"Group {gi} (n={len(groups[gi])})",
                markerfacecolor=cmap(gi),
                markeredgecolor='white',
                markersize=8
            )
            for gi in range(n_groups)
        ]
        
        if highlight_medoids:
            handles.append(
                plt.Line2D(
                    [0], [0],
                    marker='o',
                    linestyle='',
                    label='Medoids',
                    markerfacecolor='none',
                    markeredgecolor='black',
                    markersize=10,
                    markeredgewidth=2
                )
            )
        
        ax.legend(
            handles=handles,
            title=f"Leiden {cluster_id} — Groups (≤5)",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.
        )
        
        ax.set_title(
            f"UMAP Projection — Cluster {cluster_id} Groups",
            fontsize=14,
            pad=15
        )
        ax.set_xlabel("UMAP-1", fontsize=12)
        ax.set_ylabel("UMAP-2", fontsize=12)
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"   💾 Saved to {save_path}")
        
        plt.show()
    
    def plot_embeddings(
        self,
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        title: str = "UMAP Projection",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """Generic UMAP plot for any embeddings.
        
        Args:
            embeddings: High-dimensional embeddings
            labels: Optional labels for coloring
            title: Plot title
            save_path: Path to save figure
            figsize: Figure size
        """
        # Use cosine metric for dense embeddings
        reducer = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric='cosine',
            random_state=self.random_state
        )
        coords = reducer.fit_transform(embeddings)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if labels is not None:
            unique_labels = np.unique(labels)
            cmap = mpl.cm.get_cmap("tab10", len(unique_labels))
            scatter = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=labels,
                cmap=cmap,
                s=20,
                alpha=0.7
            )
            plt.colorbar(scatter, ax=ax, label="Labels")
        else:
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                s=20,
                alpha=0.7,
                c='steelblue'
            )
        
        ax.set_title(title, fontsize=14, pad=15)
        ax.set_xlabel("UMAP-1", fontsize=12)
        ax.set_ylabel("UMAP-2", fontsize=12)
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   💾 Saved to {save_path}")
        
        plt.show()