---
title: Different Type of Learning in ML
author: Aman Pandey
date: "10/20/2025"
description: Every type of learning which exists in ML a small try to keep them listed together.
editor: visual
categories: [deep-learning]
---

# Taxonomy of Learning Types in AI/ML

A structured, near-exhaustive classification of AI and ML paradigms, covering supervision, structure,  uncertainty, optimization, and emerging trends. Includes aliases, overlaps, and representative methods.

## Core Supervision Paradigms

**Methods defined by feedback and label availability.**

### Supervised Learning: Predicts outputs from labeled data.
Classification, regression  
Algorithms: SVM, Random Forests, DNNs, Transformers


### Unsupervised Learning: Finds structure without labels.
Clustering (K-means, DBSCAN), density estimation  
Dimensionality reduction (PCA, t-SNE, UMAP)


### Semi-Supervised Learning: Combines labeled and unlabeled data.
Label propagation, pseudo-labeling, co-training


### Self-Supervised Learning: Learns representations via pretext tasks.
Masked prediction (BERT), contrastive pretraining (SimCLR, MoCo)


### Weakly Supervised Learning: Uses noisy or incomplete labels.
Multi-label learning (MLL), one-class classification (OCC), positive-unlabeled learning (PUL)  
Algorithms: Snorkel, anomaly detection, PU-SVM


### Reinforcement Learning (RL): Learns via environment rewards.
Model-free (Q-learning, PPO, SAC), model-based   
Hierarchical RL, multi-agent RL, inverse RL, imitation learning  
 
---


## Geometric and Structural Learning
**Methods for non-Euclidean or structured data (umbrella: Geometric Deep Learning).**

### Graph-Based Learning: Operates on graph structures.
- Graph Neural Networks (GNNs): GCN, GAT, GraphSAGE, GIN, Graph Transformer   
- Message Passing Neural Networks (MPNNs)    
- Graph embedding, node classification, link prediction   
- Graph Representation Learning: Node2Vec, DeepWalk, GNN-based embeddings



### Manifold Learning: Captures data on continuous manifolds.
Isomap, Locally Linear Embedding (LLE), Laplacian Eigenmaps, Diffusion Maps  
UMAP for Dimensionality reduction.


### Group/Equivariant Networks: Preserves symmetries (e.g., rotations).
G-CNNs, SE(3)/E(3)-equivariant networks, steerable CNNs   


### Graph Signal Processing: Analyzes signals on graphs.
Spectral methods, wavelet transforms on graphs


### Structured Prediction:
Conditional Random Fields (CRFs), Structured SVM, Sequence labeling, parsing


---

## Probabilistic and Bayesian Approaches
**Methods modeling uncertainty and probability distributions.**

### Probabilistic Graphical Models (PGMs):   
Bayesian Networks (directed), Markov Random Fields (undirected), CRFs, HMMs


### Bayesian Learning:
Bayesian Neural Networks (BNNs), Gaussian Processes (GPs)    
Approximate inference: Variational Inference (VI), MCMC, Expectation-Maximization (EM)   


### Probabilistic Programming: Automates inference workflows.
Tools: Stan, Pyro, Edward


### Latent Variable Models:
Probabilistic Matrix Factorization, Factor Analysis



## Generative and Energy-Based Models
**Methods for data generation and likelihood modeling.**

### Generative Models:
- Autoregressive (PixelRNN, Transformer decoders)  
- Variational Autoencoders (VAEs)
- Generative Adversarial Networks (GANs)
- Normalizing Flows, Diffusion/Score-Based Models


### Energy-Based Models (EBMs):
Boltzmann Machines, Deep Belief Networks 
Score matching, energy-guided sampling

---

## Transfer, Adaptation, and Data-Efficient Learning
**Methods for reusing or adapting knowledge across tasks or domains.**

### Transfer Learning: Pretrain → finetune.
**Domain adaptation (supervised/unsupervised)**


### Multi-Task Learning: Shares representations across tasks.
- Meta-Learning: Learns how to learn new tasks.
- Few-shot, one-shot, zero-shot learning   
Algorithms: MAML, Reptile


### Continual/Lifelong/Incremental Learning: Adapts without forgetting.
Regularization-based (EWC), rehearsal-based


### Transductive Learning: Infers specific unlabeled instances.
Graph-based transductive methods


---

## Multi-Modal and Cross-Domain Learning
**Methods integrating diverse data types or domains.**

### Multi-Modal Learning: Combines data types (e.g., text, images).
Models: CLIP, DALL-E, multimodal Transformers


### Cross-Domain Learning: Adapts across different data distributions.
Domain generalization, cross-modal alignment


---

## Representation, Metric, and Contrastive Learning
**Methods for learning data embeddings or distances.**

### Representation Learning: Extracts general-purpose features.
Autoencoders, word2vec, BERT-style models, Collaborative filtering in recommendation system


### Contrastive Learning: Optimizes similarity-based objectives.
InfoNCE, SimCLR, MoCo


### Metric Learning: Learns distance metrics.
Triplet loss, Siamese Networks, N-pair loss


### Information-Theoretic Methods: Maximizes mutual information.
InfoMax, Barlow Twins


---

## Kernel and Classical Statistical Learning
**Traditional methods rooted in statistical theory.**

### Kernel Methods: Maps data to high-dimensional spaces.
SVM, Kernel PCA, Kernel Ridge Regression, RKHS theory


### Statistical Learning Theory: Analyzes generalization.
PAC learning, VC dimension, empirical risk minimization (ERM)


### Regularization Frameworks:
L1/L2 regularization, dropout


---

## Optimization and Algorithmic Variants
**Techniques for training models efficiently.**

### Gradient-Based Optimization:
SGD, Adam, RMSprop, second-order methods (L-BFGS)


### Convex vs Non-Convex Optimization:
Proximal methods, ADMM


### Meta-Optimization: Optimizes hyperparameters.
Bayesian Optimization, grid search


### Evolutionary/Neuroevolution:
Genetic algorithms, NEAT


### Swarm Intelligence:
Particle Swarm Optimization (PSO), Ant Colony Optimization


---


## Robustness, Safety, Privacy, and Fairness
**Methods addressing ethical and operational reliability.**


### Adversarial Training/Robustness:
Adversarial examples, certified defenses


### Causal Learning and Inference:
Structural Causal Models (SCMs), do-calculus, causal discovery


### Federated/Distributed Learning:
FedAvg, decentralized training


### Privacy-Preserving ML:
Differential privacy, secure multi-party computation


### Fairness-Aware Learning:
Bias mitigation, demographic parity


### Interpretability/Explainable AI (XAI):
SHAP, LIME, attention visualization


---


## Sparse, Low-Rank, and Signal Processing
**Methods for efficient data representation.**

### Sparse Learning:
LASSO, compressed sensing, LORA, Q-LORA


### Dictionary Learning/Sparse Coding:
K-SVD, sparse autoencoders

---



## Hardware-Inspired and Edge Learning
**Methods leveraging specialized hardware or constraints.**

### Edge ML: Learning on resource-constrained devices.
Edge inference, edge training, model compression (pruning, quantization)


### Neuromorphic Learning: Mimics brain-like computation.
Spiking Neural Networks (SNNs), event-driven processing


### Quantum ML: Leverages quantum computing.
Quantum kernel methods, variational quantum circuits


---


## Topological and Symbolic Learning
**Advanced methods for complex data or reasoning.**

### Topological Data Analysis (TDA):
Persistent homology, Mapper algorithm


### Causal Representation Learning:
Disentangled representations with causal structure


### Neuro-Symbolic Learning:
Inductive Logic Programming (ILP), hybrid neural-symbolic models


### Symbolic Regression:
Genetic programming for equation discovery


---


## Societal and Application-Specific Paradigms
**Methods driven by societal impact or domain needs.**

### AI for Scientific Discovery:
Hypothesis generation (e.g., AlphaFold, materials discovery)


### Value Alignment/Moral Learning:
RLHF, constitutional AI


### Human-in-the-Loop Learning:
Interactive ML, human-AI collaboration

---



## Special Settings and Practical Modes
**Operational modes for learning systems.**

### Online/Streaming Learning:
Real-time updates, concept drift handling


### Batch/Mini-Batch Learning:
Standard training on fixed datasets


### Active Learning:
Uncertainty sampling, how much model is uncertain in its prediction


### Curriculum/Self-Paced Learning:
Gradual task complexity


### Ensemble Methods:
Bagging (Random Forests), boosting (XGBoost), stacking


---


## Aliases and Overlaps

Geometric Deep Learning ≈ Graph/M Manifold/Equivariant Learning  
Graph Neural Networks ≈ Message Passing Neural Networks  
Contrastive Learning ≈ Metric/Representation Learning (similar objectives)  
Domain Adaptation ⊂ Transfer Learning  
Continual Learning ≈ Lifelong/Incremental Learning  
Probabilistic Modeling ≈ Generative Modeling (likelihood focus)  
Weak Supervision ≈ Noisy-Label Learning  

---
