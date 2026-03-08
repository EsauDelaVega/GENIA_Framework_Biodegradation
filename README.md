[README (1).md](https://github.com/user-attachments/files/23812543/README.1.md)
# GENIA Framework v2.0

**Genomically and Environmentally Networked Intelligent Assemblies**

A comprehensive computational framework for rational design of synthetic microbial communities for multi-pollutant biodegradation.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Framework Architecture](#framework-architecture)
- [Detailed Usage](#detailed-usage)
- [Input Data Format](#input-data-format)
- [Output Files](#output-files)
- [Performance Metrics](#performance-metrics)
- [Citation](#citation)
- [License](#license)

---

## 🔬 Overview

GENIA is a machine learning-guided framework for designing synthetic microbial communities (SynComs) that efficiently degrade multiple persistent environmental pollutants. The framework integrates:

- **High-throughput genomic analysis** with intelligent feature extraction
- **Metabolic network modeling** via bipartite strain-metabolite graphs
- **Graph Neural Networks (GATs)** for community-level predictions
- **Ensemble machine learning** combining Random Forest and GNN approaches
- **Optimization algorithms** for identifying optimal consortia

**Publication:** De la Vega-Camarillo et al. (2025). "Machine Learning-Guided Synthetic Microbial Communities Enable Functional and Sustainable Degradation of Persistent Environmental Pollutants." *Environmental Science & Technology*.

---

## ✨ Key Features

### Version 2.0 Enhancements

- ✅ **Complete Graph Attention Network (GAT) implementation** using PyTorch Geometric
- ✅ **Bipartite strain-metabolite graph construction** with biological interpretation
- ✅ **Node2Vec embeddings** for genomic feature representation
- ✅ **Hybrid Random Forest + GNN pipeline** for improved predictions
- ✅ **Rigorous cross-validation** with LOOCV and multiple performance metrics
- ✅ **Feature selection** using RFECV to prevent overfitting
- ✅ **Modular architecture** with clear separation of concerns
- ✅ **Comprehensive documentation** and type hints throughout

### Core Capabilities

1. **Genomic Feature Extraction**: Transforms raw genomic data into biologically-informed feature space
2. **Metabolic Network Integration**: Constructs comprehensive strain-metabolite interaction graphs
3. **Redundancy Reduction**: Identifies non-redundant, functionally diverse core communities
4. **Predictive Modeling**: 
   - Phase 1: Random Forest for strain-level predictions
   - Phase 2: GAT for community-level interactions
5. **Community Optimization**: Identifies optimal consortia for target degradation tasks

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- (Optional) CUDA-capable GPU for faster GNN training

### Step 1: Clone the Repository

```bash
git clone https://github.com/[your-lab]/GENIA.git
cd GENIA
```

### Step 2: Create Virtual Environment

#### Using conda (recommended):

```bash
conda create -n genia python=3.9
conda activate genia
```

#### Using venv:

```bash
python -m venv genia_env
source genia_env/bin/activate  # On Windows: genia_env\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install PyTorch Geometric (if not automatically installed)

For CPU only:
```bash
pip install torch-geometric
```

For CUDA (GPU):
```bash
# Replace cu118 with your CUDA version (e.g., cu116, cu117, cu121)
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### Verification

```python
python -c "import torch; import torch_geometric; print('✓ Installation successful')"
```

---

## 🎯 Quick Start

### Basic Usage

```python
import pandas as pd
import numpy as np
from genia_improved import GENIAFramework

# Load your data
genomic_data = pd.read_csv('genomic_features.csv', index_col=0)
phenotypic_data = pd.read_csv('degradation_data.csv', index_col=0)

# Initialize GENIA
genia = GENIAFramework(
    genomic_data=genomic_data,
    phenotypic_data=phenotypic_data,
    output_dir='./results'
)

# Run complete pipeline
results = genia.run_complete_pipeline(
    num_community_members=9,
    n_selected_features=89,
    use_gnn=True
)

# Access results
optimal_community = results['optimal_community']
predicted_performance = results['predicted_performance']

print(f"Optimal community: {optimal_community}")
print(f"Predicted degradation: {predicted_performance:.2f}%")
```

### With Example Data

```python
from genia_improved import main

# Run with built-in example data
main()
```

This will generate synthetic data and demonstrate the complete workflow.

---

## 🏗️ Framework Architecture

```
GENIA v2.0 Architecture
│
├─ MODULE 1: Genomic Feature Extraction
│  ├─ GenomicFeatureExtractor
│  │  ├─ Presence/absence encoding
│  │  ├─ Copy number transformation (log1p)
│  │  ├─ Shannon diversity calculation
│  │  ├─ Variance filtering
│  │  └─ Feature scaling
│  └─ Recursive Feature Elimination (RFECV)
│
├─ MODULE 2: Bipartite Graph Construction
│  ├─ BipartiteGraphBuilder
│  │  ├─ Strain nodes (bacterial isolates)
│  │  ├─ Metabolite nodes (contaminants + intermediates)
│  │  ├─ Production edges (strain → metabolite)
│  │  └─ Consumption edges (metabolite → strain)
│  └─ PyTorch Geometric conversion
│
├─ MODULE 3: Node2Vec Embeddings
│  ├─ Node2VecEmbedder
│  │  ├─ Strain co-occurrence graph construction
│  │  ├─ Random walk generation
│  │  └─ Embedding learning (128-dim)
│  └─ Fallback: Spectral embedding
│
├─ MODULE 4: Graph Attention Network
│  ├─ GATCommunityPredictor (PyTorch)
│  │  ├─ GAT Layer 1: 8 heads × 64 dims
│  │  ├─ GAT Layer 2: 4 heads × 32 dims
│  │  ├─ GAT Layer 3: 1 head × 16 dims
│  │  ├─ Global pooling
│  │  └─ FC layers → Multi-task output (3 pollutants)
│  └─ Attention weight extraction
│
├─ MODULE 5: Hybrid Prediction
│  ├─ HybridCommunityPredictor
│  │  ├─ Phase 1: Random Forest (strain-level)
│  │  │  └─ LOOCV with performance metrics
│  │  ├─ Phase 2: GAT (community-level)
│  │  │  └─ Early stopping with patience
│  │  └─ Ensemble prediction
│  └─ Greedy community optimization
│
└─ MODULE 6: GENIA Orchestrator
   └─ GENIAFramework
      ├─ Pipeline coordination
      ├─ Results aggregation
      └─ Output management
```

---

## 📊 Detailed Usage

### 1. Genomic Feature Extraction

```python
from genia_improved import GenomicFeatureExtractor

# Initialize
extractor = GenomicFeatureExtractor(
    genomic_data=your_genomic_df,
    feature_config={
        'use_presence_absence': True,
        'use_copy_numbers': True,
        'use_shannon_diversity': True,
        'variance_threshold': 0.01,
        'scale_features': True
    }
)

# Extract features
all_features = extractor.extract_features()

# Select important features
selected_features = extractor.select_important_features(
    phenotypic_data=your_phenotype_df,
    n_features=89
)

# Get feature importance
importance = extractor.feature_importance
```

### 2. Bipartite Graph Construction

```python
from genia_improved import BipartiteGraphBuilder

# Build graph
graph_builder = BipartiteGraphBuilder(
    strain_features=selected_features,
    phenotypic_data=phenotypic_data
)

# Create NetworkX graph
nx_graph = graph_builder.build_graph()

# Convert to PyTorch Geometric
pyg_data = graph_builder.to_pytorch_geometric()

# Visualize graph statistics
print(f"Nodes: {nx_graph.number_of_nodes()}")
print(f"Edges: {nx_graph.number_of_edges()}")
```

### 3. Node2Vec Embeddings

```python
from genia_improved import Node2VecEmbedder

# Generate embeddings
embedder = Node2VecEmbedder(
    strain_features=selected_features,
    embedding_dim=128
)

embeddings = embedder.generate_embeddings(
    walk_length=20,
    num_walks=100,
    p=1.0,
    q=0.5
)

# Access strain co-occurrence graph
cooccurrence_graph = embedder.strain_graph
```

### 4. Hybrid Prediction Model

```python
from genia_improved import HybridCommunityPredictor

# Initialize predictor
predictor = HybridCommunityPredictor(
    strain_features=selected_features,
    phenotypic_data=phenotypic_data,
    node2vec_embeddings=embeddings
)

# Train Phase 1 (Random Forest)
rf_metrics = predictor.train_phase1_random_forest(n_estimators=100)

# Train Phase 2 (GNN)
gnn_metrics = predictor.train_phase2_gnn(
    bipartite_graph=graph_builder,
    epochs=500,
    learning_rate=0.001,
    patience=50
)

# Predict specific community
community = ['Strain_01', 'Strain_05', 'Strain_12', ...]
prediction = predictor.predict_community(community)

# Find optimal community
optimal, performance = predictor.find_optimal_community(
    num_members=9,
    search_iterations=100
)
```

---

## 📥 Input Data Format

### Genomic Data (required)

CSV file with strains as rows, genes/features as columns:

```csv
,Gene_001,Gene_002,Gene_003,...
Strain_01,2,0,1,...
Strain_02,1,3,0,...
Strain_03,0,1,2,...
```

**Format requirements:**
- Index: Strain identifiers (strings)
- Columns: Gene/enzyme names (strings)
- Values: Gene copy numbers (integers ≥ 0) or presence/absence (0/1)

### Phenotypic Data (required)

CSV file with strains as rows, pollutants as columns:

```csv
,atrazine,pfoa,lignin
Strain_01,75.3,82.1,65.4
Strain_02,45.2,55.8,72.3
Strain_03,88.9,91.2,80.5
```

**Format requirements:**
- Index: Same strain identifiers as genomic data
- Columns: Pollutant names (strings)
- Values: Degradation rates/efficiencies (float, 0-100%)

### Optional: Metabolite Properties

CSV file with metabolite physicochemical properties (for enhanced metabolite node features):

```csv
metabolite,molecular_weight,logP,aromatic_rings,functional_groups
Atrazine,215.68,2.61,1,5
DEA,187.63,1.51,1,4
PFOA,414.07,-1.0,0,8
```

---

## 📤 Output Files

GENIA generates the following outputs in the specified `output_dir`:

### 1. **selected_features.csv**
Selected genomic features (strains × 89 features) after RFECV

### 2. **feature_importance.csv**
Importance scores for each selected feature

### 3. **node2vec_embeddings.csv**
128-dimensional embeddings for each strain

### 4. **optimal_community.json**
```json
{
  "strains": ["Strain_05", "Strain_12", "Strain_23", ...],
  "predicted_performance": 92.5
}
```

### 5. **rf_model.pkl**
Trained Random Forest model (pickle format)

### 6. **performance_metrics.json**
```json
{
  "atrazine": {
    "r2": 0.71,
    "rmse": 8.3,
    "mae": 6.1,
    "pearson_r": 0.85
  },
  "pfoa": {...},
  "lignin": {...},
  "overall": {...}
}
```

---

## 📈 Performance Metrics

GENIA reports comprehensive validation metrics:

### Cross-Validation Metrics

- **R² (Coefficient of Determination)**: Proportion of variance explained
- **RMSE (Root Mean Squared Error)**: Average prediction error
- **MAE (Mean Absolute Error)**: Robust to outliers
- **Pearson correlation (r)**: Linear association strength
- **Spearman correlation (ρ)**: Rank-order association

### Model Interpretation

- **Feature importance**: Contribution of each genomic feature
- **Attention weights**: GAT learned relationships between nodes
- **Learning curves**: Training vs. validation performance
- **Residual analysis**: Model assumptions validation

---

## 🔧 Configuration Options

### GenomicFeatureExtractor Config

```python
feature_config = {
    'use_presence_absence': True,      # Binary encoding
    'use_copy_numbers': True,          # Log-transformed counts
    'use_shannon_diversity': True,     # Diversity index
    'variance_threshold': 0.01,        # Low-variance filter
    'scale_features': True             # StandardScaler
}
```

### GNN Training Config

```python
gnn_config = {
    'input_dim': 217,                  # 89 features + 128 embeddings
    'hidden_dims': [64, 32, 16],       # GAT layer dimensions
    'attention_heads': [8, 4, 1],      # Heads per layer
    'output_dim': 3,                   # Number of pollutants
    'dropout': 0.3,                    # Dropout probability
    'epochs': 500,                     # Maximum epochs
    'learning_rate': 0.001,            # AdamW learning rate
    'patience': 50                     # Early stopping patience
}
```

---

## 🧪 Testing

Run unit tests:

```bash
pytest tests/ -v --cov=genia_improved
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run linting
flake8 genia_improved.py

# Run type checking
mypy genia_improved.py
```

---

## 📚 Citation

If you use GENIA in your research, please cite:

```bibtex
@article{delavega2025genia,
  title={Machine Learning-Guided Synthetic Microbial Communities Enable 
         Functional and Sustainable Degradation of Persistent Environmental Pollutants},
  author={De la Vega-Camarillo, Esaú; Arreola-Vargas, Jorge; Mathur, Saurav; Santos, Joshua; Antony-Babu, Sanjay; and Shim, Won Bo},
  journal={BiorXiv},
  year={2025},
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Texas A&M University** - Plant Pathology & Microbiology Department
- **Dr. Sanjay Antony-Babu's & Dr. Won Bo Shim's Laboratories**
- **PyTorch Geometric** developers for excellent graph learning tools
- **NetworkX** and **scikit-learn** communities

---

## 📞 Contact

- **Lead Developer**: Esaú De la Vega-Camarillo
- **Principal Investigator**: Dr. Sanjay Antony-Babu & Dr. Won Bo Shim
- **Institution**: Texas A&M University
- **Email**: [esaudelavega@tamu.edu]
- **GitHub**: [https://github.com/EsauDelaVega/GENIA]

---

## 🐛 Known Issues & Troubleshooting

### Issue: PyTorch Geometric installation fails

**Solution**: Install PyTorch first, then PyTorch Geometric:
```bash
pip install torch
pip install torch-geometric -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)")+cpu.html
```

### Issue: Node2Vec takes too long

**Solution**: Reduce `num_walks` or use the fallback spectral embedding:
```python
# In Node2VecEmbedder.generate_embeddings()
num_walks=50  # instead of 100
```

### Issue: Out of memory during GNN training

**Solution**: Reduce batch size or hidden dimensions:
```python
gnn_config = {
    'hidden_dims': [32, 16, 8],  # Smaller dimensions
    'attention_heads': [4, 2, 1]  # Fewer heads
}
```

---

## 🗺️ Roadmap

### Version 2.1 (Planned)
- [ ] Multi-objective optimization for consortia design
- [ ] Integration with metabolic modeling (COBRA)
- [ ] Web-based interactive interface
- [ ] Automated genome annotation pipeline

### Version 3.0 (Future)
- [ ] Transfer learning from pre-trained models
- [ ] Active learning for experimental design
- [ ] Real-time monitoring integration
- [ ] Cloud deployment support

---

## 📊 Example Results

### Optimal 9-member Community for Multi-pollutant Degradation

```
Selected Strains:
1. Pseudomonas sp. A
2. Burkholderia sp. C
3. Sphingomonas sp. F
4. Rhodococcus sp. H
5. Arthrobacter sp. J
6. Streptomyces sp. L
7. Mycobacterium sp. N
8. Bacillus sp. P
9. Microbacterium sp. R

Predicted Performance:
- Atrazine: 92.1 ± 2.8%
- PFOA: 94.3 ± 3.1%
- Lignin: 91.7 ± 4.3%

Model Performance (LOOCV):
- Overall R² = 0.71 ± 0.05
- Overall RMSE = 8.24 ± 0.31
```

---

**Last Updated**: November 2025  
**Version**: 2.0  
**Status**: Active Development
