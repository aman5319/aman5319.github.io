### **1. Partitioning-based**
- K-Means  
- Bisecting K-Means  
- K-Medoids / PAM (Partitioning Around Medoids)  
- CLARA (Clustering Large Applications)  
- Fuzzy C-Means (Soft clustering)  

---

### **2. Hierarchical**
- Agglomerative Clustering  
- Ward Hierarchical Clustering  
- BIRCH  
- CURE (Clustering Using Representatives)  
- ROCK (Robust Clustering using Links)  

---

### **3. Density-Based**
- DBSCAN  
- HDBSCAN  
- OPTICS  
- DENCLUE (DENsity-based CLUstEring)  

---

### **4. Model-Based / Probabilistic**
- Gaussian Mixture Models (GMM)  
- Hidden Markov Model clustering (for sequential data)  
- Latent Dirichlet Allocation (for topic modeling, can be considered clustering)  

---

### **5. Graph / Spectral**
- Spectral Clustering  
- Community detection algorithms (e.g., Louvain, Girvan-Newman)  

---

### **6. Other / Miscellaneous**
- Affinity Propagation  
- Mean Shift  
- Self-Organizing Maps (SOM)  
- Agglomerative Fuzzy Clustering  
- Subspace clustering (e.g., CLIQUE, PROCLUS)  
- Co-clustering / Biclustering (e.g., for matrices or gene expression data)  



Perfect. I’ll create a **master table** that combines all the clustering algorithms we’ve discussed, along with **parameters, geometry, cluster size preference, inductive vs. transductive, scalability, and typical use cases**. I’ll organize it so it’s easy to reference in your article.

---

| **Algorithm**                                     | **Type**                     | **Key Parameters**                                               | **Geometry / Distance Metric**          | **Cluster Size Preference** | **Inductive / Transductive** | **Scalability**                         | **Typical Use Cases**                             |
| ------------------------------------------------- | ---------------------------- | ---------------------------------------------------------------- | --------------------------------------- | --------------------------- | ---------------------------- | --------------------------------------- | ------------------------------------------------- |
| **K-Means**                                       | Partitioning                 | Number of clusters (k)                                           | Euclidean (distances between points)    | Even                        | Inductive                    | Very large n_samples, medium n_clusters | General-purpose, flat geometry, simple clusters   |
| **Bisecting K-Means**                             | Partitioning / Hierarchical  | Number of clusters (k)                                           | Euclidean                               | Even                        | Inductive                    | Very large n_samples, medium n_clusters | Hierarchical general-purpose clustering           |
| **K-Medoids / PAM**                               | Partitioning                 | Number of clusters (k)                                           | Any pairwise distance                   | Even                        | Inductive                    | Medium n_samples                        | Robust to outliers, small datasets                |
| **CLARA**                                         | Partitioning                 | Number of clusters, sample size                                  | Any pairwise distance                   | Even                        | Inductive                    | Large n_samples                         | Large dataset approximation of K-Medoids          |
| **Fuzzy C-Means**                                 | Partitioning                 | Number of clusters, fuzziness (m)                                | Euclidean                               | Overlapping                 | Inductive                    | Medium n_samples                        | Soft clustering, overlapping clusters             |
| **Agglomerative Clustering**                      | Hierarchical                 | Number of clusters or distance threshold, linkage type, distance | Any pairwise distance                   | Flexible                    | Transductive                 | Large n_samples                         | Many clusters, possibly connectivity constraints  |
| **Ward Hierarchical**                             | Hierarchical                 | Number of clusters or distance threshold                         | Euclidean                               | Even                        | Transductive                 | Large n_samples                         | Minimizes intra-cluster variance                  |
| **BIRCH**                                         | Hierarchical / Tree-based    | Branching factor, threshold, optional global clusterer           | Euclidean                               | Flexible                    | Inductive                    | Large n_samples & n_clusters            | Large dataset clustering, data reduction          |
| **DBSCAN**                                        | Density-Based                | ε (neighborhood size), min_samples                               | Distance between nearest points         | Uneven                      | Transductive                 | Very large n_samples, medium n_clusters | Non-flat clusters, outlier detection              |
| **HDBSCAN**                                       | Density-Based / Hierarchical | min_cluster_size, min_samples                                    | Distance between nearest points         | Uneven, variable density    | Transductive                 | Large n_samples                         | Non-flat clusters, hierarchical density, outliers |
| **OPTICS**                                        | Density-Based                | min_samples                                                      | Distances between points                | Uneven, variable density    | Transductive                 | Very large n_samples                    | Variable density clusters, outlier detection      |
| **DENCLUE**                                       | Density-Based                | Kernel bandwidth, density threshold                              | Euclidean                               | Flexible                    | Inductive                    | Medium n_samples                        | Arbitrary-shaped clusters, density-based          |
| **Gaussian Mixture Models (GMM)**                 | Model-Based                  | Number of components, covariance type                            | Mahalanobis distance                    | Flexible                    | Inductive                    | Not scalable                            | Density estimation, soft probabilistic clusters   |
| **Spectral Clustering**                           | Graph-Based                  | Number of clusters, affinity type                                | Graph distance (e.g., nearest-neighbor) | Even                        | Transductive                 | Medium n_samples, small n_clusters      | Non-flat geometry, connectivity-based clusters    |
| **Affinity Propagation**                          | Other                        | Damping, preference                                              | Graph distance (similarity)             | Uneven                      | Inductive                    | Not scalable                            | Many clusters, irregular shapes                   |
| **Mean Shift**                                    | Other / Density-Based        | Bandwidth                                                        | Euclidean                               | Uneven                      | Inductive                    | Not scalable                            | Arbitrary-shaped clusters, non-flat geometry      |
| **Self-Organizing Maps (SOM)**                    | Other / Neural               | Grid size, learning rate                                         | Euclidean                               | Flexible                    | Inductive                    | Medium n_samples                        | Visualization, topology-preserving clustering     |
| **CURE**                                          | Hierarchical                 | Number of clusters, shrink factor                                | Euclidean                               | Uneven                      | Transductive                 | Medium n_samples                        | Large, non-spherical clusters                     |
| **ROCK**                                          | Hierarchical                 | Number of clusters, similarity threshold                         | Link-based similarity                   | Uneven                      | Transductive                 | Medium n_samples                        | Categorical data clustering                       |
| **CLIQUE / PROCLUS**                              | Subspace / High-Dimensional  | Density threshold, subspace size                                 | Euclidean / Manhattan                   | Flexible                    | Inductive                    | Medium n_samples, high-dimensional      | High-dimensional data, subspace clusters          |
| **Co-Clustering / Biclustering**                  | Other                        | Number of row/column clusters                                    | Euclidean / similarity                  | Flexible                    | Inductive / Transductive     | Medium n_samples                        | Matrix data, gene expression, recommendation data |
| **Community Detection (Louvain / Girvan-Newman)** | Graph-Based                  | Resolution, modularity                                           | Graph distance                          | Flexible                    | Transductive                 | Medium n_samples                        | Network/community clustering                      |
