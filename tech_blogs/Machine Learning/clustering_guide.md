
---
title: "Choosing a Clustering Algorithm"
author: "Aman Pandey"
date: "3/12/2026"
editor: visual
categories: [machine-learning]
image: "images/cluster.png"
description: "A dive into how to choose a clustering algorithm for your problems"
---


# Choosing a Clustering Algorithm: A scikit-learn-Only Decision Guide

scikit-learn's own [clustering page](https://scikit-learn.org/stable/modules/clustering.html) is the right reference to build this on: it's the one place that's actually maintained, benchmarked, and consistent. The problem isn't the page itself; it's that it presents 12 algorithms side by side with no order of operations. This guide fixes that: same 12 algorithms, but organized as a sequence of questions instead of a flat table.

## The running example

Say you've run topic extraction over a large pile of documents and ended up with, e.g., 40,000 raw topic phrases, many of them near-duplicates in meaning: "password reset," "forgot my password," "reset login credentials." You embed each phrase with a sentence embedding model:

```python
embeddings = model.encode(topic_phrases)
print(embeddings.shape)
# (40000, 128)
```

40,000 rows, 128 columns. Now you want to cluster these embeddings so every group of near-duplicate phrases collapses into one canonical topic name: "Password Reset," "Billing Dispute," and so on.

That single `.shape` call already tells you two things before you've picked anything: 40K samples rules out the algorithms that don't scale past a few thousand points, and 128 features is enough to make raw Euclidean distance start behaving oddly (more on that below). We'll come back to this scenario for every pattern below.

## The 12 algorithms in scope

Everything in this guide is one of these, nothing else. If you've seen a clustering algorithm mentioned elsewhere that isn't on this list, it's not in scikit-learn's `cluster` module (or `mixture`, for Gaussian Mixture).

| Algorithm | sklearn class |
|---|---|
| K-Means | `cluster.KMeans` |
| Mini-Batch K-Means | `cluster.MiniBatchKMeans` |
| Bisecting K-Means | `cluster.BisectingKMeans` |
| Affinity Propagation | `cluster.AffinityPropagation` |
| Mean Shift | `cluster.MeanShift` |
| Spectral Clustering | `cluster.SpectralClustering` |
| Agglomerative Clustering (Ward/complete/average/single) | `cluster.AgglomerativeClustering` |
| DBSCAN | `cluster.DBSCAN` |
| HDBSCAN | `cluster.HDBSCAN` |
| OPTICS | `cluster.OPTICS` |
| BIRCH | `cluster.Birch` |
| Gaussian Mixture Model | `mixture.GaussianMixture` |

Note: "Ward hierarchical clustering" isn't a separate class; it's `AgglomerativeClustering` with `linkage='ward'`. sklearn's own comparison table lists it as its own row because the linkage choice changes behavior that much, but it's one estimator with a parameter, not two algorithms.

## The 15 decision patterns, ordered by importance

Ordered so that the questions which eliminate whole families of algorithms come first, and the questions that are more about practical deployment fit come last.

### 1. Do you know the number of clusters (k) upfront?

The single biggest fork. `KMeans`, `MiniBatchKMeans`, `BisectingKMeans`, `SpectralClustering`, and `GaussianMixture` all require `n_clusters` as a parameter. `DBSCAN`, `HDBSCAN`, `OPTICS`, `MeanShift`, and `AffinityPropagation` discover the count from the data.

**Intuition:** In the topic-phrase example, you almost never know upfront how many canonical topics exist in 40,000 raw phrases, since that's usually the whole point of running the clustering. This alone pushes you toward `HDBSCAN` or `OPTICS` over `KMeans`, before you've even thought about anything else.

### 2. What shape are your clusters: convex or arbitrary?

`KMeans`, `MiniBatchKMeans`, `BisectingKMeans`, and `GaussianMixture` (with default covariance) assume roughly spherical, similarly-sized clusters. `DBSCAN`, `HDBSCAN`, `OPTICS`, `SpectralClustering`, and `AgglomerativeClustering` can follow arbitrary shapes.

**Intuition:** Semantically similar phrases in embedding space don't form neat spheres: "password reset" and "account recovery" might sit in a long, curved region of the embedding space rather than a tight ball. Forcing `KMeans` onto that shape will cut clusters in the wrong place, even with the right k.

### 3. Do outliers need to stay unclustered, or must every point get a label?

`DBSCAN`, `HDBSCAN`, and `OPTICS` explicitly label sparse points as noise (`-1`) instead of forcing them into the nearest cluster. `KMeans`, `GaussianMixture`, `AgglomerativeClustering`, `SpectralClustering`, `BIRCH`, and `BisectingKMeans` assign every point somewhere, no exceptions.

**Intuition:** Some of those 40,000 phrases will be genuinely one-off: garbled extraction, or a topic that appears exactly once. You don't want those forced into "Password Reset" just because it's the nearest centroid; that pollutes the canonical name for everyone else in the cluster. This is a real vote for density-based methods.

### 4. Is cluster density roughly uniform, or does it vary a lot?

Plain `DBSCAN` assumes one global density threshold (`eps`) and struggles when some clusters are dense and others are sparse. `HDBSCAN` was built specifically to handle varying density without you having to pick a single `eps`.

**Intuition:** "Password Reset" might have hundreds of near-duplicate phrasings (very dense), while "Refund for Duplicate Charge" might have three (sparse). A single `eps` that works for one will merge or shatter the other. This is usually the deciding factor between `DBSCAN` and `HDBSCAN` in practice, and `HDBSCAN` wins more often than not for text-embedding tasks like this.

### 5. What counts as a real cluster? (minimum viable cluster size)

Every density-based method here has some notion of a minimum group size before it counts as a cluster at all: `HDBSCAN` and `OPTICS` take `min_cluster_size`, `DBSCAN` takes `min_samples`. This isn't only a technical dial, it's a business decision about how much evidence you need before you're willing to give a group a name.

**Intuition:** A canonical topic backed by three phrases is probably real; a "cluster" made of one stray phrase is more likely a parsing artifact than an actual topic. Deciding "groups under N phrases get folded into a Misc bucket instead of getting their own name" is worth doing on purpose, tied directly to `min_cluster_size`, rather than accepting whatever the default gives you and being surprised later by a canonical topic named after a typo.

### 6. How big is the dataset, and does that rule anything out?

- `AffinityPropagation`: O(n²) time and memory, realistically capped at a few thousand points.
- Plain `AgglomerativeClustering` without a connectivity constraint: also expensive at scale, though sklearn implements it efficiently for moderate n.
- `SpectralClustering`: needs an eigendecomposition of an n×n matrix, fine for small/medium n, but painful past that.
- `KMeans`, `MiniBatchKMeans`, `BisectingKMeans`, `BIRCH`, `DBSCAN`, `HDBSCAN`: built to handle large n; `MiniBatchKMeans` and `BIRCH` are the ones explicitly designed for very large datasets.

**Intuition:** 40,000 points immediately rules out `AffinityPropagation` as a serious option: that's already millions of pairwise comparisons before you've clustered anything. `HDBSCAN` and `MiniBatchKMeans` are both comfortable here; `SpectralClustering` is borderline and would need the affinity matrix built carefully (sparse, k-NN graph) to stay feasible.

### 7. Do you need one flat partition, or a full hierarchy?

`AgglomerativeClustering` gives you a dendrogram you can cut at any level. `BisectingKMeans` produces a rough hierarchy as a side effect of how it splits. `BIRCH` builds an internal tree too, though it's mainly used for data reduction rather than for reading off multiple hierarchy levels. Everything else (`KMeans`, `DBSCAN`, `SpectralClustering`, `GaussianMixture`, `AffinityPropagation`) gives you one flat answer.

**Intuition:** You might eventually want "Account Access Issues" as a broad category with "Password Reset" and "2FA Problems" nested underneath it. If that's a real requirement (not just nice-to-have), `AgglomerativeClustering` is worth the extra cost even at 40K points; if a flat list of canonical topics is genuinely enough, don't pay for a hierarchy you won't use.

### 8. Do you need hard labels, or soft/probabilistic ones?

`GaussianMixture` gives every point a probability of belonging to each cluster. Every other algorithm on this list gives a single hard label.

**Intuition:** A phrase like "can't log in after resetting my password" genuinely straddles two canonical topics. If you want to say "70% Password Reset, 30% Login Issues" instead of forcing a single bucket, `GaussianMixture` is the only option here, but note it also assumes roughly elliptical clusters, so you're trading shape flexibility for the soft labels.

### 9. Is the data a fixed batch, or does it keep arriving?

`MiniBatchKMeans` and `BIRCH` (via `partial_fit`) are built to update incrementally as new data shows up. The rest expect the full dataset up front and require a full re-fit to incorporate new points.

**Intuition:** If new support tickets keep landing daily and you want topic clusters to update without reclustering all 40,000+ phrases from scratch every night, that pushes hard toward `MiniBatchKMeans`. `HDBSCAN`'s clustering quality has to be weighed against the fact that it doesn't have a natural incremental-update path in sklearn.

### 10. Do you need to keep clusters fresh as new data keeps arriving?

This is a different question from pattern 9. Pattern 9 asks whether a single model can be *updated* incrementally. This one asks what happens to your *existing clusters* over weeks and months, as the data drifts and genuinely new groups show up that didn't exist when you first fit the model.

Two related facts matter here. First, sklearn's own documentation splits every algorithm into **inductive** (fit once, then `.predict()` labels a brand-new point without touching the training data again) or **transductive** (only has labels for the exact points it was fit on). `KMeans`, `MiniBatchKMeans`, `BisectingKMeans`, `MeanShift`, `AffinityPropagation`, `BIRCH`, and `GaussianMixture` are inductive. `DBSCAN`, `HDBSCAN`, `OPTICS`, `AgglomerativeClustering`, and `SpectralClustering` are transductive: giving a genuinely new point a label means either a full refit or a manual workaround, such as assigning it to the nearest existing cluster by distance.

Second, even an inductive model goes stale on its own. Assigning new points to old centroids doesn't discover new clusters that didn't exist before, and it doesn't notice that an old cluster has drifted or quietly split into two. A pattern that holds up in practice:

- Use an inductive method's `.predict()` for cheap, real-time assignment of new points to existing clusters between refits.
- On a schedule that matches your drift rate (nightly, weekly, whatever fits), refit from scratch on the accumulated data, including a density-based pass like `HDBSCAN`, to check whether genuinely distinct new groups have emerged that the inductive model would otherwise have silently absorbed into an existing cluster.
- Reconcile the new cluster set against the old one by matching new centroids to old ones (nearest-centroid matching, or the Hungarian algorithm for an optimal one-to-one match), so "Password Reset" keeps meaning the same thing across refreshes instead of quietly becoming cluster #7 one week and cluster #12 the next.
- Route brand-new clusters that don't match anything old to a human for naming, rather than assuming the pipeline should name them on its own.

**Intuition:** If new tickets keep landing daily, real-time `.predict()` with `MiniBatchKMeans` keeps things responsive, but only a periodic full re-cluster will catch the day a genuinely new topic (say, a product launch generating a wave of identical complaints) actually shows up as its own group, instead of getting smeared across whichever existing cluster happens to be closest.


 >  #### River 
    >To continuously find new topics as they come in, we can use the package [river](https://github.com/online-ml/river). It contains several clustering models that can create new clusters as new data comes in.


### 11. What's the dimensionality of your feature space?

Distance and density both get less meaningful as dimensions increase (the curse of dimensionality). `BIRCH` in particular is known to scale poorly once you're past roughly 20 features; sklearn's own docs recommend switching to `MiniBatchKMeans` at that point. Density-based methods (`DBSCAN`, `HDBSCAN`, `OPTICS`) also get shakier in very high dimensions because "density" stops being a clean concept.

**Intuition:** 128 dimensions is already past BIRCH's comfort zone: that alone removes it from contention for this scenario. It's not yet catastrophic for `HDBSCAN`, but if you'd embedded at 768 or 1536 dimensions (common for larger embedding models), running PCA or UMAP down to 20-50 dimensions before clustering would likely help more than switching algorithms.

### 12. What's your underlying data type, and does that change the metric?

All 12 sklearn algorithms here expect numeric feature vectors or a precomputed distance/affinity matrix. There's no dedicated categorical-clustering class in sklearn's cluster module; mixed or categorical data has to be numerically encoded first, or passed in as a precomputed distance matrix to one of the algorithms that accepts it.

**Intuition:** Text embeddings are numeric by construction, so this pattern mostly resolves itself here. The real decision it forces is the *metric* (next pattern), not the algorithm family.

### 13. Do you need a custom distance metric, or is Euclidean fine?

`KMeans`, `MiniBatchKMeans`, and `BisectingKMeans` are built around Euclidean distance (they optimize toward a mean). `AgglomerativeClustering`, `DBSCAN`, `OPTICS`, `SpectralClustering`, and `AffinityPropagation` all accept a custom metric or a precomputed distance/affinity matrix.

**Intuition:** Text embeddings are usually compared with cosine similarity, not raw Euclidean distance. You can sidestep this for `KMeans` by L2-normalizing the embeddings first (which makes Euclidean distance rank-equivalent to cosine), or you can just use an algorithm that takes cosine distance natively. See the dedicated section below for how to choose a metric in general.

### 14. Do you need the same result on every run?

`KMeans`, `MiniBatchKMeans`, and `GaussianMixture` depend on random initialization; different seeds can give different results unless you fix `random_state`. `DBSCAN`, `HDBSCAN`, `OPTICS`, and `AgglomerativeClustering` are deterministic given the same parameters (DBSCAN has minor, well-documented order-dependence at cluster borders, but it's not seed-driven).

**Intuition:** If "Password Reset" needs to mean the same cluster every time you rerun the pipeline, say because downstream dashboards reference it by ID, reproducibility stops being a nice-to-have. Worth fixing `random_state` regardless of which algorithm you land on, and worth weighting deterministic methods slightly higher if this matters a lot.

### 15. Do you need to explain the result to a non-technical stakeholder?

Centroid-based methods (`KMeans`, `MiniBatchKMeans`, `BisectingKMeans`) are the easiest to explain: "this phrase is closest to the centroid of cluster 3." `AgglomerativeClustering` is a close second since a dendrogram is intuitive to walk through. Density- and graph-based methods (`HDBSCAN`, `SpectralClustering`, `AffinityPropagation`) are harder to justify in a business review because there's no single simple "distance to X" story.

**Intuition:** If you're handing "Password Reset" clusters to a support-ops team who will ask "why did ticket #4821 end up in this bucket," a centroid or a dendrogram gives you a one-line answer. `HDBSCAN`'s density-based logic is real and often more accurate, but it's a harder story to tell in a meeting.

## Common distance metrics, and how to decide

Here are the ones that actually show up in production, with the kind of dataset each one gets used on rather than a textbook definition.

- **Euclidean (L2)**: straight-line distance. Real use: color quantization in image compression, where K-Means clusters the RGB values of every pixel to reduce a photo to a fixed palette; also clustering standardized sensor or measurement data where raw magnitude carries meaning.
- **Manhattan / Cityblock (L1)**: sum of absolute coordinate differences. Real use: clustering grid-constrained movement data, like warehouse robot paths, where actual travel distance runs along axes rather than diagonally; also a common choice for sparse, high-dimensional count data (word counts, event counts), since it's less dominated by a handful of large values than Euclidean is.
- **Cosine distance**: angle between two vectors, ignoring length. Real use: clustering sentence or document embeddings, exactly the running example here, and clustering user taste vectors in recommendation systems, e.g. grouping listeners by the relative mix of genres they play rather than by total listening volume.
- **Haversine (great-circle) distance**: real use: clustering GPS coordinates, such as grouping ride-share pickup requests, delivery addresses, or retail store locations by geographic proximity. Plain Euclidean distance on raw latitude/longitude is misleading here, since a degree of longitude covers a different physical distance depending on latitude; Haversine (or projecting to a flat coordinate system first) accounts for that.
- **Jaccard distance**: overlap between two sets, ignoring order and count. Real use: clustering users by the sets of products they've purchased or pages they've visited (market-basket-style clustering), and near-duplicate document detection using sets of word shingles.
- **Hamming distance**: number of differing positions between two equal-length sequences. Real use: clustering binary or one-hot encoded categorical profiles, and grouping hashed fingerprints produced by locality-sensitive hashing.
- **Precomputed / custom**: any distance you can compute yourself (Mahalanobis for correlated numeric features, e.g. clustering financial transactions where amount and frequency move together, or edit distance for clustering raw strings like product SKUs or misspelled search queries) can be handed to most of these algorithms as a full n×n distance matrix via `metric='precomputed'`. This is the escape hatch when none of the named metrics fit.

**How to decide, quickly:**
1. Is your data an embedding (text, image, audio)? Cosine, almost by default.
2. Is your data geographic coordinates? Haversine, or project to a flat coordinate system first.
3. Is your data binary, one-hot, or a set of items (purchases, tags, pages visited)? Jaccard for sets, Hamming for fixed-length codes.
4. Are your features on very different scales or units? Standardize first regardless of metric; Euclidean is then usually fine.
5. Do a few extreme values risk dominating the distance calculation? Manhattan over Euclidean.
6. None of the above fit? Compute a domain-specific distance yourself and pass it in as `metric='precomputed'`.

**Back to the running example:** text embeddings point straight at cosine. `KMeans`, `MiniBatchKMeans`, and Ward linkage all optimize around a literal mean, which only lines up with cosine distance once you L2-normalize every embedding vector first: after normalizing, ranking points by Euclidean distance gives the same order as ranking by cosine distance, which is why this is the standard workaround. If you'd rather not normalize, `AgglomerativeClustering` (average or complete linkage), `DBSCAN`, and `OPTICS` can take `metric='cosine'` directly, though it's worth checking each class's current docs, since a few of these use tree-based backends internally that sometimes require `metric='precomputed'` instead of naming the metric directly.

## Quick reference: pattern → algorithm

| Pattern | Points toward |
|---|---|
| Don't know k | HDBSCAN, OPTICS, DBSCAN, MeanShift, AffinityPropagation |
| Convex, similar-size clusters | KMeans, MiniBatchKMeans, BisectingKMeans, GaussianMixture |
| Arbitrary/non-convex shape | DBSCAN, HDBSCAN, OPTICS, SpectralClustering, AgglomerativeClustering |
| Outliers must stay unclustered | DBSCAN, HDBSCAN, OPTICS |
| Need to set a minimum group size before naming a cluster | HDBSCAN/OPTICS (`min_cluster_size`), DBSCAN (`min_samples`) |
| Density varies across clusters | HDBSCAN (not plain DBSCAN) |
| Very large n | MiniBatchKMeans, BIRCH (if features under ~20), HDBSCAN |
| Small n only | AffinityPropagation, SpectralClustering, plain AgglomerativeClustering |
| Need full hierarchy | AgglomerativeClustering, BisectingKMeans |
| Need soft/probabilistic labels | GaussianMixture |
| Streaming / incremental updates to one model | MiniBatchKMeans, BIRCH |
| Need to label brand-new points without a full refit | Inductive only: KMeans, MiniBatchKMeans, BisectingKMeans, MeanShift, AffinityPropagation, BIRCH, GaussianMixture |
| High-dimensional raw features (over 20) | Avoid BIRCH; reduce dims first or use MiniBatchKMeans / SpectralClustering |
| Need non-Euclidean or precomputed distance | AgglomerativeClustering, DBSCAN, OPTICS, SpectralClustering, AffinityPropagation |
| Need reproducible results every run | AgglomerativeClustering, DBSCAN/HDBSCAN, OPTICS |
| Need to explain results simply | KMeans, MiniBatchKMeans, BisectingKMeans, AgglomerativeClustering |

## The full catalog, by family (sklearn only)

### Centroid-based
- **K-Means** (`cluster.KMeans`): assigns points to the nearest of k centroids, then updates centroids iteratively. Fast and simple, assumes spherical, similar-sized clusters.
- **Mini-Batch K-Means** (`cluster.MiniBatchKMeans`): K-Means run on small random batches instead of the full dataset each iteration. Built for large or streaming data, at a small cost to accuracy.
- **Bisecting K-Means** (`cluster.BisectingKMeans`): repeatedly splits one cluster into two using K-Means until k clusters exist. Produces a rough hierarchy and tends to give more even cluster sizes than plain K-Means.

### Density-based
- **DBSCAN** (`cluster.DBSCAN`): groups points that are densely packed together and marks sparse points as noise. No k required, but assumes roughly uniform density and needs `eps`/`min_samples` tuned.
- **HDBSCAN** (`cluster.HDBSCAN`): extends DBSCAN to handle clusters of varying density, without needing a single global `eps`. Generally the more robust default over plain DBSCAN.
- **OPTICS** (`cluster.OPTICS`): builds a reachability ordering instead of committing to one density threshold, so you can extract clusters at multiple density levels from a single run.
- **Mean Shift** (`cluster.MeanShift`): shifts candidate centroids toward the nearest mode (density peak) until convergence. No k required, but not scalable to large n and sensitive to the bandwidth parameter.

### Hierarchical
- **Agglomerative Clustering** (`cluster.AgglomerativeClustering`): starts with every point as its own cluster and repeatedly merges the closest pair, bottom-up, until one cluster remains; you then cut the resulting dendrogram at whatever level gives you the k (or distance threshold) you want. What "closest" means is entirely defined by the **linkage** you pick, and that choice changes behavior as much as switching algorithms would:

  - **Ward**: merges whichever pair increases total within-cluster variance the least (the same objective KMeans optimizes, but reached hierarchically). Euclidean-only. Produces the most even, compact, spherical-ish clusters of the four; the natural default for numeric data with roughly globular structure.
  - **Complete (maximum) linkage**: cluster-to-cluster distance is the *farthest* pair of points between them. Produces tight, compact clusters and avoids "chaining," but one distant outlier in either cluster can block a merge that should otherwise happen.
  - **Average linkage**: cluster-to-cluster distance is the *mean* of all cross-cluster pairwise distances. A middle ground between complete and single, and the one that plays nicest with non-Euclidean metrics (cosine, Manhattan, precomputed); usually the right pick whenever Ward isn't an option.
  - **Single linkage**: cluster-to-cluster distance is the *closest* pair of points between them. Can trace out long, thin, non-convex shapes the other three would never find, but is notorious for "chaining": a handful of connecting points can fuse two clusters that should have stayed separate, especially with noisy data.

  **Quick pick:** Euclidean data, want compact and evenly-sized groups → Ward. Non-Euclidean metric (e.g., cosine on embeddings), still want compact groups → Average. Genuinely elongated or chain-like structure, and the data is clean → Single. Want compact clusters but Ward's variance criterion doesn't apply to your metric → Complete.
- **BIRCH** (`cluster.Birch`): builds a compact tree summary of the data first, then clusters the summary. Designed for large datasets, but scales poorly past roughly 20 features.

### Probabilistic
- **Gaussian Mixture Model** (`mixture.GaussianMixture`): fits a mixture of Gaussians via Expectation-Maximization, giving soft, probabilistic cluster assignments. Requires k (or a model-selection step to choose it) and assumes elliptical clusters.

### Graph-based
- **Spectral Clustering** (`cluster.SpectralClustering`): builds a similarity graph, then clusters the eigenvectors of its Laplacian. Handles non-convex shapes well; requires k and doesn't scale past small/medium n.
- **Affinity Propagation** (`cluster.AffinityPropagation`): passes messages between points until a subset emerges as cluster exemplars. No k required, but O(n²) cost limits it to small datasets.

## Practical tips that matter more than the algorithm choice

- **Scale your features** before using any distance-based method (KMeans, DBSCAN, Agglomerative). For embeddings specifically, L2-normalizing rows is usually the right move if you intend cosine-like comparisons.
- **Validate with more than one metric**: silhouette score, Davies-Bouldin, and a 2D projection (PCA/UMAP) each catch different failure modes.
- **Try KMeans first as a cheap baseline**, even if you're fairly sure it isn't the final answer; it's fast and gives you something to sanity-check fancier methods against.
- **For HDBSCAN/DBSCAN, tune `min_samples`/`eps` using a k-distance plot** rather than guessing: sklearn's own DBSCAN docs point to this as the standard approach.
- **Check `n_features` before reaching for BIRCH.** Past ~20 dimensions, sklearn's own guidance is to prefer MiniBatchKMeans.

## After the clustering: naming each cluster

Clustering only gets you groups. Turning each group into one usable name is a separate step, and it's the actual deliverable in the running example. Three common approaches, roughly in increasing order of cost and quality:

- **Nearest-to-center phrase**: pick whichever real phrase sits closest to the cluster's centroid (or medoid, for algorithms without a true centroid) and use it as the name directly. Cheap and always grounded in something a person actually wrote, but that phrase can be an oddly specific stand-in for the group ("reset my acct pw plz" standing in for "Password Reset").
- **Top terms**: extract the most frequent or highest-TF-IDF-weighted words across all members of the cluster and stitch them into a label. More representative of the group as a whole, but can produce label soup ("reset password login account forgot") rather than a clean name.
- **LLM summarization**: pass a sample of the cluster's members to a language model and ask for one short canonical name. Produces the most readable names, but can hallucinate a label that doesn't quite match what's actually in the cluster, especially for noisy or mixed clusters, so it's worth spot-checking a sample against the members before trusting it at scale.

A reasonable default: nearest-to-center or top-terms for a first pass, with LLM summarization reserved for clusters that clear the minimum viable size from pattern 5, since that's where getting the name right actually matters.

## One-paragraph version

For the 40K-topic-phrase scenario: you don't know k, clusters vary in density, and you want outliers left out; that combination points straight at **HDBSCAN**. If you needed incremental updates as new tickets arrive instead, you'd trade some cluster-shape flexibility for **MiniBatchKMeans**. If stakeholders needed a simple "why is this here" answer and near-spherical topic groups were good enough, **KMeans** would be the pragmatic choice even though it's not the best technical fit. That's the real shape of this decision: not "which of the 12 is best," but "which two or three fit this specific combination of constraints, and which constraint are you willing to trade off."
