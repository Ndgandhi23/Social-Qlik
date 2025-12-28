#!/usr/bin/env python
"""Quick start example for interest clustering."""

from src.clustering.pipeline import ClusteringPipeline
from src.utils.data_loader import load_sample_data
from src.utils.metrics import evaluate_clustering


def main():
    """Run a simple clustering example."""
    print("🚀 Interest-Based Clustering Demo\n")
    
    # Load sample data
    print("📊 Loading sample data...")
    texts, labels = load_sample_data()
    print(f"   ✓ Loaded {len(texts)} texts across {len(set(labels))} categories\n")
    
    # Initialize pipeline
    print("⚙️  Initializing clustering pipeline...")
    pipeline = ClusteringPipeline(
        leiden_resolution=1.0,
        group_capacity=5,
        random_state=42
    )
    print("   ✓ Pipeline ready\n")
    
    # Run clustering
    print("🔄 Running clustering pipeline...")
    results = pipeline.fit(texts, labels)
    
    # Print results
    print("\n" + "="*60)
    print("📈 CLUSTERING RESULTS")
    print("="*60)
    print(f"Total items: {results['n_items']}")
    print(f"Categories: {results['n_categories']}")
    print(f"Leiden clusters: {results['n_leiden_clusters']}")
    print(f"Final groups: {results['n_groups']}")
    print(f"\nGroup sizes:")
    print(f"  Min: {results['group_size_min']}")
    print(f"  Max: {results['group_size_max']}")
    print(f"  Mean: {results['group_size_mean']:.2f}")
    print("="*60 + "\n")
    
    # Evaluate quality
    print("📊 Evaluating clustering quality...")
    metrics = evaluate_clustering(
        interest_vectors=pipeline.interest_vectors_,
        group_assignments=pipeline.group_global_,
        leiden_labels=pipeline.leiden_labels_
    )
    
    print("\n" + "="*60)
    print("📊 QUALITY METRICS")
    print("="*60)
    print(f"Within-group similarity: {metrics['avg_intra_similarity']:.3f}")
    print(f"Between-group distance: {metrics['avg_inter_distance']:.3f}")
    print(f"Silhouette (groups): {metrics['silhouette_groups']:.3f}")
    print(f"Silhouette (Leiden): {metrics['silhouette_leiden']:.3f}")
    print(f"Balance ratio: {metrics['balance_balance_ratio']:.3f}")
    print("="*60 + "\n")
    
    # Show example groups
    print("👥 Example Groups:\n")
    for group_id in range(min(3, results['n_groups'])):
        members = pipeline.get_group_members(group_id)
        print(f"Group {group_id} ({len(members)} members):")
        for idx, text, label in members:
            print(f"  • {text[:50]}... [{label}]")
        print()
    
    # Visualize
    print("🎨 Generating visualizations...")
    try:
        import os
        os.makedirs('output', exist_ok=True)
        
        pipeline.visualize_clusters(
            save_path='output/leiden_clusters.png',
            figsize=(12, 8)
        )
        print("   ✓ Saved Leiden cluster visualization")
        
        # Visualize first cluster's groups
        if results['n_leiden_clusters'] > 0:
            pipeline.visualize_groups(
                cluster_id=0,
                highlight_medoids=True,
                save_path='output/cluster_0_groups.png',
                figsize=(10, 7)
            )
            print("   ✓ Saved group visualization for cluster 0")
        
        print(f"\n💾 Visualizations saved to 'output/' directory")
        
    except Exception as e:
        print(f"   ⚠️  Visualization skipped: {e}")
    
    print("\n✨ Demo complete!")
    print("\nNext steps:")
    print("  • Check out notebooks/01_clustering_demo.ipynb")
    print("  • Read docs/algorithms.md for implementation details")
    print("  • Customize config.yaml for your use case")
    print("  • Run tests with: pytest tests/")


if __name__ == "__main__":
    main()