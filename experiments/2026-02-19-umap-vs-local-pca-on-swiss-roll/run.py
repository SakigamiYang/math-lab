from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import umap
from mathlab.repro import env_info_dict
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class Config:
    n_samples: int = 4000
    noise: float = 0.05
    random_state: int = 42

    # local PCA
    k_neighbors: int = 30
    pca_max_dim: int = 10

    # embedding dims to compare
    embed_dims: tuple[int, ...] = (1, 2, 3, 5, 10)

    # UMAP params
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "euclidean"

    # metrics
    metric_k: int = 15  # for trustworthiness & continuity


def continuity(X: np.ndarray, Y: np.ndarray, k: int) -> float:
    """
    Continuity metric (0..1). Measures how well neighbors in embedding Y
    are preserved in original space X.

    Implementation: compute kNN in X and Y; continuity is average neighbor overlap.
    (A simplified, interpretable version; good for comparisons across dims.)
    """
    nn_x = NearestNeighbors(n_neighbors=k + 1).fit(X)
    nn_y = NearestNeighbors(n_neighbors=k + 1).fit(Y)
    idx_x = nn_x.kneighbors(X, return_distance=False)[:, 1:]
    idx_y = nn_y.kneighbors(Y, return_distance=False)[:, 1:]

    # overlap size per point
    overlap = [len(set(idx_x[i]).intersection(idx_y[i])) for i in range(X.shape[0])]
    return float(np.mean(np.array(overlap) / k))


def local_pca_spectrum(X: np.ndarray, k: int, max_dim: int, rng: np.random.Generator) -> np.ndarray:
    """
    For a subset of points, compute PCA eigenvalue spectrum on each local neighborhood,
    then average the normalized eigenvalues.
    """
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    neigh_idx = nn.kneighbors(X, return_distance=False)[:, 1:]

    # sample points to keep runtime reasonable
    m = min(800, X.shape[0])
    sample = rng.choice(X.shape[0], size=m, replace=False)

    spectra = []
    for i in sample:
        Xi = X[neigh_idx[i]]  # (k, D)
        Xi = Xi - Xi.mean(axis=0, keepdims=True)
        pca = PCA(n_components=min(max_dim, Xi.shape[1]))
        pca.fit(Xi)
        ev = pca.explained_variance_
        ev = ev / (ev.sum() + 1e-12)
        if len(ev) < max_dim:
            ev = np.pad(ev, (0, max_dim - len(ev)), constant_values=0.0)
        spectra.append(ev[:max_dim])

    return np.mean(np.stack(spectra, axis=0), axis=0)


def spectral_gap_dim(avg_spectrum: np.ndarray) -> dict:
    """
    Return spectral gaps and a heuristic intrinsic dimension:
    argmax gap in log-spectrum differences.
    """
    eps = 1e-12
    logv = np.log(avg_spectrum + eps)
    gaps = logv[:-1] - logv[1:]
    d_hat = int(np.argmax(gaps) + 1)  # gap after d_hat-th component
    return {"gaps": gaps.tolist(), "d_hat": d_hat}


def save_fig_scatter_2d(path: Path, Y: np.ndarray, color: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(Y[:, 0], Y[:, 1], s=4, c=color)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    cfg = Config()
    here = Path(__file__).resolve().parent
    out_dir = here / "artifacts"
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.random_state)

    # 1) data
    X, t = make_swiss_roll(n_samples=cfg.n_samples, noise=cfg.noise, random_state=cfg.random_state)
    # normalize for stability
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

    # 2) local PCA spectrum + intrinsic dim heuristic
    avg_spec = local_pca_spectrum(X, k=cfg.k_neighbors, max_dim=cfg.pca_max_dim, rng=rng)
    gap_info = spectral_gap_dim(avg_spec)

    # save spectrum table
    df_spec = pd.DataFrame(
        {
            "component": np.arange(1, cfg.pca_max_dim + 1),
            "avg_explained_variance_ratio": avg_spec,
        }
    )
    df_spec.to_csv(tab_dir / "local_pca_avg_spectrum.csv", index=False)

    # save spectrum plot
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(df_spec["component"], df_spec["avg_explained_variance_ratio"], marker="o")
    plt.yscale("log")
    plt.xlabel("PCA component (local neighborhood)")
    plt.ylabel("Avg explained variance ratio (log scale)")
    plt.title(f"Local PCA spectrum (k={cfg.k_neighbors}); d_hat={gap_info['d_hat']}")
    plt.tight_layout()
    plt.savefig(fig_dir / "local_pca_spectrum.png", dpi=180)
    plt.close()

    # 3) baseline: global PCA embeddings
    results = []
    for d in cfg.embed_dims:
        pca = PCA(n_components=min(d, X.shape[1]))
        t0 = perf_counter()
        Y = pca.fit_transform(X)
        dt = perf_counter() - t0

        tw = trustworthiness(X, Y, n_neighbors=cfg.metric_k)
        cont = continuity(X, Y, k=cfg.metric_k)

        results.append(
            {
                "method": "PCA",
                "embed_dim": d,
                "trustworthiness_k": cfg.metric_k,
                "trustworthiness": float(tw),
                "continuity": float(cont),
                "fit_seconds": float(dt),
            }
        )

        if d == 2:
            save_fig_scatter_2d(fig_dir / "pca_2d.png", Y, t, "PCA 2D embedding (colored by swiss-roll t)")

    # 4) UMAP embeddings
    for d in cfg.embed_dims:
        reducer = umap.UMAP(
            n_components=d,
            n_neighbors=cfg.umap_n_neighbors,
            min_dist=cfg.umap_min_dist,
            metric=cfg.umap_metric,
            random_state=cfg.random_state,
        )
        t0 = perf_counter()
        Y = reducer.fit_transform(X)
        dt = perf_counter() - t0

        tw = trustworthiness(X, Y, n_neighbors=cfg.metric_k)
        cont = continuity(X, Y, k=cfg.metric_k)

        results.append(
            {
                "method": "UMAP",
                "embed_dim": d,
                "trustworthiness_k": cfg.metric_k,
                "trustworthiness": float(tw),
                "continuity": float(cont),
                "fit_seconds": float(dt),
                "umap_n_neighbors": cfg.umap_n_neighbors,
                "umap_min_dist": cfg.umap_min_dist,
                "umap_metric": cfg.umap_metric,
            }
        )

        if d == 2:
            save_fig_scatter_2d(
                fig_dir / "umap_2d.png",
                Y,
                t,
                f"UMAP 2D (n_neighbors={cfg.umap_n_neighbors}, min_dist={cfg.umap_min_dist})",
            )

    df = pd.DataFrame(results).sort_values(["method", "embed_dim"])
    df.to_csv(tab_dir / "embedding_metrics.csv", index=False)

    # plot metric curves
    plt.figure()
    for method in ["PCA", "UMAP"]:
        sub = df[df["method"] == method].sort_values("embed_dim")
        plt.plot(sub["embed_dim"], sub["trustworthiness"], marker="o", label=f"{method} trustworthiness")
    plt.xlabel("Embedding dimension")
    plt.ylabel(f"Trustworthiness (k={cfg.metric_k})")
    plt.title("Trustworthiness vs embedding dimension")
    plt.tight_layout()
    plt.savefig(fig_dir / "trustworthiness_vs_dim.png", dpi=180)
    plt.close()

    plt.figure()
    for method in ["PCA", "UMAP"]:
        sub = df[df["method"] == method].sort_values("embed_dim")
        plt.plot(sub["embed_dim"], sub["continuity"], marker="o", label=f"{method} continuity")
    plt.xlabel("Embedding dimension")
    plt.ylabel(f"Continuity (k={cfg.metric_k})")
    plt.title("Continuity vs embedding dimension")
    plt.tight_layout()
    plt.savefig(fig_dir / "continuity_vs_dim.png", dpi=180)
    plt.close()

    # metadata
    metadata = {
        "experiment": here.name,
        "config": asdict(cfg),
        "local_pca": {
            "avg_spectrum": avg_spec.tolist(),
            "spectral_gap": gap_info,
        },
        "outputs": {
            "tables": [
                "tables/local_pca_avg_spectrum.csv",
                "tables/embedding_metrics.csv",
            ],
            "figures": [
                "figures/local_pca_spectrum.png",
                "figures/pca_2d.png",
                "figures/umap_2d.png",
                "figures/trustworthiness_vs_dim.png",
                "figures/continuity_vs_dim.png",
            ],
        },
        "env": env_info_dict(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Done.")
    print(f"- {tab_dir / 'local_pca_avg_spectrum.csv'}")
    print(f"- {tab_dir / 'embedding_metrics.csv'}")
    print(f"- {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
