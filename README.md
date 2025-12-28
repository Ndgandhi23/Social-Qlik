# Interest-Based Clustering System

A production-ready clustering pipeline that groups items based on shared interests using semantic embeddings, Leiden community detection, and capacitated k-medoids clustering.

## 🎯 What This Does

Takes text descriptions with category labels and creates **balanced groups of similar items** (max 5 members per group).

**Example Input:**
```python
texts = ["basketball practice", "calculus homework", "photography workshop"]
labels = ["Sports / Fitness", "Academic / Study", "Hobbies / Creative"]
```

**Example Output:**
- 4 Leiden clusters based on interest similarity
- 15 balanced groups (≤5 members each)
- UMAP visualizations showing cluster structure

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd interest-clustering-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Demo
```bash
python examples/quick_start.py
```

This will:
1. Load 180+ sample activities across 4 categories
2. Generate MPNet embeddings (384-dim)
3. Run Leiden clustering on interest similarity graph
4. Form balanced groups using k-medoids
5. Save visualizations to `output/`

### Use in Your Code
```python
from src.clustering.pipeline import ClusteringPipeline

# Initialize
pipeline = ClusteringPipeline(
    leiden_resolution=1.0,  # Higher = more clusters
    group_capacity=5        # Max group size
)

# Run clustering
results = pipeline.fit(texts, labels)

# Get group members
members = pipeline.get_group_members(group_id=0)
for idx, text, label in members:
    print(f"{text} [{label}]")

# Visualize
pipeline.visualize_clusters(save_path='clusters.png')
pipeline.visualize_groups(cluster_id=0, save_path='groups.png')
```

## 📊 How It Works
```
Text Input
    ↓
MPNet Embeddings (384-dim)
    ↓
Binary Interest Vectors
    ↓
k-NN Graph Construction
    ↓
Leiden Community Detection
    ↓
Capacitated K-Medoids (≤5 per group)
    ↓
Balanced Groups
```

### Key Algorithms

1. **MPNet Embeddings** - Converts text to semantic vectors using `sentence-transformers/all-mpnet-base-v2`
2. **Interest Vectors** - Binary representations of category membership
3. **Leiden Clustering** - Graph-based community detection on Jaccard similarity
4. **Capacitated K-Medoids** - Balanced grouping with size constraints

## 📁 Project Structure
```
interest-clustering-project/
├── src/
│   ├── embeddings/          # MPNet + interest vectors
│   ├── clustering/          # Leiden + k-medoids + pipeline
│   ├── visualization/       # UMAP plotting
│   └── utils/              # Data loading + metrics
├── tests/                  # Unit tests
├── examples/               # Demo scripts
├── notebooks/              # Jupyter notebooks
├── .vscode/               # VS Code configuration
├── requirements.txt       # Dependencies
├── config.yaml           # Configuration
└── README.md
```

## 🔧 Configuration

Edit `config.yaml` to customize behavior:
```yaml
leiden:
  resolution: 1.0        # Cluster granularity (0.5-2.0)
  n_neighbors: 15        # Graph connectivity

kmedoids:
  capacity: 5            # Max group size
  max_iterations: 10     # Convergence iterations
```

## 🧪 Testing
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_clustering.py -v
```

## 📈 Evaluation Metrics

The pipeline computes:

- **Intra-cluster similarity** - How similar are items within groups?
- **Inter-cluster distance** - How different are groups from each other?
- **Silhouette score** - Overall clustering quality (-1 to 1)
- **Balance ratio** - Are group sizes balanced? (0 to 1)
```python
from src.utils.metrics import evaluate_clustering

metrics = evaluate_clustering(
    interest_vectors=pipeline.interest_vectors_,
    group_assignments=pipeline.group_global_,
    leiden_labels=pipeline.leiden_labels_
)

print(f"Within-group similarity: {metrics['avg_intra_similarity']:.3f}")
print(f"Silhouette score: {metrics['silhouette_groups']:.3f}")
```

## 🎨 VS Code Integration

Open the project in VS Code:
```bash
code .
```

**Features:**
- ✅ Auto-format on save (Black)
- ✅ Linting with flake8
- ✅ Integrated testing
- ✅ Debug configurations (press F5)
- ✅ Recommended extensions

## 📚 Advanced Usage

### Custom Data
```python
from src.utils.data_loader import load_sample_data

# Replace with your data
texts = ["your", "text", "data"]
labels = ["category1", "category2", "category1"]

pipeline = ClusteringPipeline()
results = pipeline.fit(texts, labels)
```

### Adjust Parameters
```python
pipeline = ClusteringPipeline(
    leiden_resolution=1.5,     # More clusters
    leiden_n_neighbors=20,     # Denser graph
    group_capacity=3,          # Smaller groups
    random_state=42
)
```

### Export Results
```python
# Get all results as dictionary
export_data = pipeline.export_results()

# Save to pickle
import pickle
with open('results.pkl', 'wb') as f:
    pickle.dump(export_data, f)
```

## 🔬 Algorithm Details

See [`docs/algorithms.md`](docs/algorithms.md) for in-depth explanations of:
- Leiden community detection
- Capacitated k-medoids
- Jaccard similarity computation
- UMAP dimensionality reduction
- Complexity analysis

## 📦 Dependencies

**Core:**
- numpy, pandas, scikit-learn
- torch, sentence-transformers
- igraph, leidenalg
- umap-learn, matplotlib

**Development:**
- pytest, black, flake8
- jupyter

See [`requirements.txt`](requirements.txt) for full list.

## 🤝 Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development setup and guidelines.

## 📄 License

MIT License - see [`LICENSE`](LICENSE) for details.

## 🙏 Acknowledgments

- **Sentence Transformers** - MPNet embeddings
- **Leiden Algorithm** - Community detection
- **UMAP** - Dimensionality reduction

## 📧 Support

- 🐛 Report bugs via GitHub Issues
- 💡 Request features via GitHub Issues
- 📖 Read the docs in `/docs`

---

**Built for scalable, production-ready interest-based clustering** 🚀
