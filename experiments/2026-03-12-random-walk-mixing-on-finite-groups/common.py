from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def total_variation_distance(p: np.ndarray, u: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - u).sum())


def mixing_curve(
    transition: np.ndarray,
    start_idx: int,
    max_steps: int,
) -> tuple[list[int], list[float]]:
    n = transition.shape[0]
    p = np.zeros(n, dtype=float)
    p[start_idx] = 1.0
    u = np.full(n, 1.0 / n, dtype=float)

    steps: list[int] = []
    tvs: list[float] = []

    for t in range(max_steps + 1):
        steps.append(t)
        tvs.append(total_variation_distance(p, u))
        p = p @ transition

    return steps, tvs


def mixing_time(
    transition: np.ndarray,
    start_idx: int,
    max_steps: int,
    threshold: float = 0.01,
) -> int | None:
    steps, tvs = mixing_curve(transition=transition, start_idx=start_idx, max_steps=max_steps)
    for t, d_tv in zip(steps, tvs, strict=False):
        if d_tv < threshold:
            return t
    return None


def save_curve_plot(
    steps: list[int],
    values: list[float],
    out_path: Path,
    title: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.plot(steps, values)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_probability_heatmap(
    transition: np.ndarray,
    start_idx: int,
    max_steps: int,
    out_path: Path,
    title: str,
) -> None:
    n = transition.shape[0]
    p = np.zeros(n, dtype=float)
    p[start_idx] = 1.0

    history = [p.copy()]
    for _ in range(max_steps):
        p = p @ transition
        history.append(p.copy())

    mat = np.array(history).T

    plt.figure(figsize=(8, 4.8))
    plt.imshow(mat, aspect="auto", origin="lower")
    plt.xlabel("step")
    plt.ylabel("state index")
    plt.title(title)
    plt.colorbar(label="probability")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
