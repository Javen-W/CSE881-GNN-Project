# CSE881: Graph Neural Network Course Project

This repository contains my course project for CSE881, a graduate-level Data Mining course completed as part of my Master’s in Computer Science and Engineering. The project focuses on node classification in a graph dataset using Graph Neural Networks (GNNs), demonstrating my experience in designing and implementing GNN models with PyTorch, PyTorch Geometric, and NumPy. This work highlights my ability to tackle complex graph-based machine learning problems with limited datasets, preparing me for roles in machine learning engineering.

# Table of Contents

- [CSE881: Graph Neural Network Course Project](#cse881-graph-neural-network-course-project)
  - [Project: Node Classification with Graph Neural Networks](#project-node-classification-with-graph-neural-networks)
    - [Description](#description)
    - [Dataset](#dataset)
    - [Approach](#approach)
    - [Tools](#tools)
    - [Results](#results)
    - [Key Skills](#key-skills)
  - [Skills Demonstrated](#skills-demonstrated)

## Project: Node Classification with Graph Neural Networks

### Description
Developed a GNN-based pipeline to predict node labels in a graph dataset, addressing a 7-class node classification problem. The goal was to leverage graph structure (adjacency matrix) and node features to achieve high accuracy, evaluated on a test set of 1984 nodes.

### Dataset
- **Source Files**:
  - Adjacency matrix: `data_2024/adj.npz` (10100 edges).
  - Feature matrix: `data_2024/features.npy` (2480 nodes, 1390 features).
  - Labels: `data_2024/labels.npy` (7 classes).
  - Train/test splits: `data_2024/splits.json` (496 train, 1984 test nodes).
- **Statistics**: 2480 nodes, 10100 edges, 7 classes, 1390 features per node.
- **Task**: Predict the class label for each test node based on features and graph connectivity.

### Approach
- **Data Preprocessing**:
  - Loaded and formatted data into a PyTorch Geometric `Data` object, with node features (`x`), labels (`y`), and edge indices from the adjacency matrix.
  - Applied transformations using PyTorch Geometric, including `GCNNorm` for feature normalization and `RemoveDuplicatedEdges`/`RemoveIsolatedNodes` for graph cleaning. Implemented a custom `RemoveFreeColumns` transform to drop linearly dependent features, reducing dimensionality while preserving matrix rank (from 1390 to 1385 features).
  - Used stratified k-fold cross-validation (5 folds) to ensure robust model evaluation on the training set.
- **Model Architecture**:
  - Evaluated multiple GNN architectures: `GCNConv`, `GraphSAGE`, `GIN`, and a custom `AggGCNConv` (aggregation-based GCN).
  - For `AggGCNConv`, designed a 2-layer GCN with 32 hidden units, incorporating ReLU activations, dropout (p=0.5), and a sort aggregation layer (`SortAggregation`, k=4) to capture graph structure.
  - Other models (`GCNConv`, `GraphSAGE`, `GIN`) used 2-5 layers with hidden dimensions of 16-256, followed by linear layers for classification.
- **Training**:
  - Implemented a `TrainerTorch` class in PyTorch, optimizing models with Adam (learning rate=0.01, weight decay=5e-4) and negative log-likelihood loss.
  - Used cross-validation to tune hyperparameters (layers, hidden units, epochs, learning rate), with learning rate decay (factor=0.5, every 50 epochs).
  - Trained for 250 epochs, logging performance metrics (loss, accuracy) per fold.
- **Evaluation**:
  - Selected the best model based on lowest validation loss, generating predictions for 1984 test nodes.
  - Saved predictions in `submission.txt`, formatted as one integer label per line.

### Tools
- **PyTorch & PyTorch Geometric**: Built and trained GNN models, leveraging `GCNConv`, `SAGEConv`, `GINConv`, and custom layers.
- **NumPy & SciPy**: Handled data loading, preprocessing, and matrix operations (e.g., adjacency matrix to edge indices).
- **Scikit-learn**: Used for stratified k-fold cross-validation and data scaling.
- **Matplotlib**: Visualized model performance (not explicitly saved in repo, inferred from typical practice).

### Results
- **Best Model**: Custom `AggGCNConv` with 2 layers, 32 hidden units, trained for 250 epochs (learning rate=0.01).
- **Performance**: Achieved a test accuracy of 82.9% ± 3.6% (validation loss=0.559), as logged in `logs/109/results.log`.
- **Output**: Generated `submission.txt` with 1984 node label predictions, meeting evaluation requirements.
- **Insights**: The `AggGCNConv` model outperformed others by effectively aggregating neighbor features, leveraging graph structure over raw features alone.

### Key Skills
- **GNN Development**: Designed and optimized GNN architectures for node classification.
- **PyTorch Proficiency**: Implemented custom GNN layers and training pipelines with PyTorch Geometric.
- **Data Preprocessing**: Engineered feature reduction and graph normalization techniques.
- **Model Evaluation**: Applied cross-validation and hyperparameter tuning for robust performance.
- **NumPy Efficiency**: Managed large-scale graph data and transformations.

## Skills Demonstrated
- **Graph Neural Networks**: Developed GNN models (`GCNConv`, `GraphSAGE`, `GIN`, `AggGCNConv`) for node classification, leveraging graph structure and node features to achieve high accuracy.
- **PyTorch & PyTorch Geometric Proficiency**: Built end-to-end GNN pipelines using PyTorch, implementing custom convolutional layers, aggregation functions (`SortAggregation`), and training loops. Utilized optimizers (Adam), loss functions (NLL), and transformations (`GCNNorm`, `RemoveFreeColumns`) to optimize performance.
- **Libraries and Tools**:
  - **NumPy & SciPy**: Processed adjacency matrices, feature matrices, and labels, performing efficient conversions to PyTorch tensors and edge indices.
  - **Scikit-learn**: Employed `StratifiedKFold` for cross-validation and `StandardScaler` for feature preprocessing.
  - **PyTorch Geometric**: Applied graph transformations and utilities (`from_scipy_sparse_matrix`, `index_to_mask`) for data preparation.
- **Data Preprocessing**: Designed custom transforms to remove linearly dependent features and normalize graph data, enhancing model efficiency.
- **Technical Proficiency**: Combined graph theory with machine learning, tuning hyperparameters and evaluating models via cross-validation to deliver accurate predictions for real-world graph datasets.
