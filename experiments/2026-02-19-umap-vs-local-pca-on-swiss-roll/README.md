# UMAP vs local PCA on Swiss roll

## Goal

This experiment investigates two related questions:

1. Can local PCA spectral gaps recover the intrinsic dimension of the Swiss roll?
2. How does UMAP behave across different embedding dimensions, measured using trustworthiness and continuity?

The Swiss roll is a classical example of a 2-dimensional nonlinear manifold embedded in 3D space.
We test whether local linear structure (via PCA) reveals a clear spectral gap corresponding to intrinsic dimension 2, and compare this with nonlinear embedding quality from UMAP.

## Mathematical setup

### Data

We generate a Swiss roll dataset:

- Ambient dimension: 3
- Intrinsic dimension: 2
- Gaussian noise added
- Standardized before analysis

### Local PCA

For each point:

- Compute k-nearest neighbors in ambient space
- Center the neighborhood
- Fit PCA
- Normalize eigenvalues

We average the local eigenvalue spectra across many sampled neighborhoods.

We estimate intrinsic dimension $\hat{d}$ as:

$$
\hat{d} = \arg \max_{i} (\log \lambda_{i} - \log \lambda_{i+1})
$$

This corresponds to the largest spectral gap in log-scale.

### Embedding Evaluation Metrics

For PCA and UMAP embeddings across multiple target dimensions $d$:

We compute:

- **Trustworthiness (k-neighbors)** \
Measures how many original neighbors remain neighbors after embedding.
- **Continuity (k-neighbors)** \
Measures how well neighbors in embedding space correspond to neighbors in original space.

## Experiment design
Discretization / sampling / parameters / metrics.

Both metrics range from 0 to 1 (higher is better).

## Experiment Design

- Sample size: 4000
- Local PCA neighborhood size: 30
- Embedding dimensions tested: 1, 2, 3, 5, 10
- UMAP parameters:
  - `n_neighbors = 15`
  - `min_dist = 0.1`
  - `metric = euclidean`

## How to run
```bash
uv run python run.py
```

## Outputs

- `artifacts/metadata.json`
- `artifacts/figures/*`
- `artifacts/tables/*`

## Results

- Local PCA spectrum should exhibit a clear spectral gap around dimension 2.
- PCA embeddings will struggle to unfold the manifold globally.
- UMAP should outperform PCA in 2D embeddings in terms of trustworthiness.
- Increasing embedding dimension should generally increase metric values.

## Notes

Potential extensions:

- Study how intrinsic dimension estimate varies with neighborhood size.
- Perform a hyperparameter sweep for UMAP.
- Compare with Isomap or diffusion maps.
- Analyze behavior under higher noise levels.
