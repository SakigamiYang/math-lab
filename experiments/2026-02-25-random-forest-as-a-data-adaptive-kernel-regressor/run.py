from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from mathlab.repro import env_info_dict
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Config:
    # Samples
    n_samples: int = 800
    n_featuers: int = 10
    n_informative: int = 5
    noise: float = 15.0
    random_state: int = 42

    # Train & Test split
    test_size: float = 0.25

    # Random forest regressor
    n_estimators: int = 200
    bootstrap: bool = False
    max_features: str = "sqrt"
    min_samples_leaf: int = 5


def rf_weights_from_leaves(rf: RandomForestRegressor, X_train: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    """
    Compute RF regression weights w(x) over training samples for each query point x.

    For each tree b:
      w_i^(b)(x) = 1{leaf_b(x_i) == leaf_b(x)} / |S_b(x)|
    Then average over trees.
    """
    estimators = rf.estimators_
    B = len(estimators)
    n_train = X_train.shape[0]
    n_query = X_query.shape[0]

    train_leaf = np.vstack([t.apply(X_train) for t in estimators])  # (B, n_train)
    query_leaf = np.vstack([t.apply(X_query) for t in estimators])  # (B, n_query)

    # Precompute leaf sizes for each tree: leaf_id -> count
    leaf_size_maps: list[dict[int, int]] = []
    for b in range(B):
        ids, counts = np.unique(train_leaf[b], return_counts=True)
        leaf_size_maps.append(dict(zip(ids.tolist(), counts.tolist(), strict=False)))

    W = np.zeros((n_query, n_train), dtype=float)

    for j in range(n_query):
        w = np.zeros(n_train, dtype=float)
        for b in range(B):
            leaf_id = int(query_leaf[b, j])
            same = (train_leaf[b] == leaf_id).astype(float)
            denom = float(leaf_size_maps[b][leaf_id])
            w += same / denom
        w /= float(B)
        W[j] = w

    # Sanity: each row should sum to 1 (up to float error)
    return W


def main() -> None:
    here = Path(__file__).resolve().parent
    out_dir = here / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()

    X, y = make_regression(
        n_samples=cfg.n_samples,
        n_features=cfg.n_featuers,
        n_informative=cfg.n_informative,
        noise=cfg.noise,
        random_state=cfg.random_state,
    )
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.random_state)

    rf = RandomForestRegressor(
        n_estimators=cfg.n_estimators,
        bootstrap=cfg.bootstrap,
        max_features=cfg.max_features,
        min_samples_leaf=cfg.min_samples_leaf,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    W = rf_weights_from_leaves(rf, X_train, X_test)
    y_recon = W @ y_train
    y_rf = rf.predict(X_test)

    max_abs_err = float(np.max(np.abs(y_recon - y_rf)))
    mean_abs_err = float(np.mean(np.abs(y_recon - y_rf)))
    row_sum_err = float(np.max(np.abs(W.sum(axis=1) - 1.0)))

    print(f"max_abs_err={max_abs_err:.3e}")
    print(f"mean_abs_err={mean_abs_err:.3e}")
    print(f"max_row_sum_err={row_sum_err:.3e}")

    metadata = {
        "experiment": here.name,
        "env": env_info_dict(),
        "config": asdict(cfg),
        "regressor_error": {
            "max_abs_err": max_abs_err,
            "mean_abs_err": mean_abs_err,
            "row_sum_err": row_sum_err,
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote: {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
