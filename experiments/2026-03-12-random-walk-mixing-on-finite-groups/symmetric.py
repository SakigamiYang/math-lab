from __future__ import annotations

from itertools import permutations
from pathlib import Path

import numpy as np
from common import (
    ensure_dir,
    mixing_curve,
    mixing_time,
    save_curve_plot,
    save_probability_heatmap,
)
from mathlab.numerics import markov_spectral_gap

Perm = tuple[int, ...]


def compose(p: Perm, q: Perm) -> Perm:
    """
    Composition p ∘ q, acting on 0..n-1:
        (p ∘ q)(i) = p[q[i]]
    """
    return tuple(p[q[i]] for i in range(len(p)))


def transposition(n: int, i: int, j: int) -> Perm:
    arr = list(range(n))
    arr[i], arr[j] = arr[j], arr[i]
    return tuple(arr)


def symmetric_elements(n: int) -> list[Perm]:
    return [tuple(p) for p in permutations(range(n))]


def build_symmetric_transition(n: int, generators: list[Perm], lazy_prob: float = 0.5) -> np.ndarray:
    elems = symmetric_elements(n)
    idx = {g: i for i, g in enumerate(elems)}
    m = len(elems)
    transition = np.zeros((m, m), dtype=float)

    for x in elems:
        i = idx[x]
        transition[i, i] += lazy_prob
        move_prob = (1.0 - lazy_prob) / len(generators)
        for s in generators:
            y = compose(s, x)
            j = idx[y]
            transition[i, j] += move_prob

    return transition


def run_symmetric_experiments(
    out_dir: Path,
    max_steps: int = 80,
) -> list[dict[str, object]]:
    figures_dir = out_dir / "figures"
    ensure_dir(figures_dir)

    configs = [
        {
            "group": "S_4_adjacent",
            "n": 4,
            "generators": [
                transposition(4, 0, 1),
                transposition(4, 1, 2),
                transposition(4, 2, 3),
            ],
        },
        {
            "group": "S_4_star",
            "n": 4,
            "generators": [
                transposition(4, 0, 1),
                transposition(4, 0, 2),
                transposition(4, 0, 3),
            ],
        },
    ]

    rows: list[dict[str, object]] = []

    for cfg in configs:
        group_name = str(cfg["group"])
        n = int(cfg["n"])
        generators = list(cfg["generators"])

        transition = build_symmetric_transition(n=n, generators=generators, lazy_prob=0.5)
        identity = tuple(range(n))
        elems = symmetric_elements(n)
        start_idx = elems.index(identity)

        steps, tvs = mixing_curve(transition=transition, start_idx=start_idx, max_steps=max_steps)
        gap = markov_spectral_gap(transition)
        t_mix = mixing_time(
            transition=transition,
            start_idx=start_idx,
            max_steps=max_steps,
            threshold=0.01,
        )

        slug = f"symmetric_{group_name}"

        save_curve_plot(
            steps=steps,
            values=tvs,
            out_path=figures_dir / f"{slug}_tv.png",
            title=f"{group_name}",
            ylabel="TV distance",
        )
        save_probability_heatmap(
            transition=transition,
            start_idx=start_idx,
            max_steps=max_steps,
            out_path=figures_dir / f"{slug}_heatmap.png",
            title=f"{group_name} heatmap",
        )

        rows.append(
            {
                "family": "symmetric",
                "group": group_name,
                "size": len(elems),
                "generators": str(generators),
                "spectral_gap": gap,
                "mixing_time_eps_0.01": t_mix,
            }
        )

    return rows
