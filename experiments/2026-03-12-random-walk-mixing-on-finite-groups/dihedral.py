from __future__ import annotations

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

# Represent D_n elements as tuples:
# ("r", k)  for rotations r^k
# ("s", k)  for reflections s r^k
Element = tuple[str, int]


def dihedral_elements(n: int) -> list[Element]:
    return [("r", k) for k in range(n)] + [("s", k) for k in range(n)]


def dihedral_multiply(a: Element, b: Element, n: int) -> Element:
    ta, ka = a
    tb, kb = b

    if ta == "r" and tb == "r":
        return "r", (ka + kb) % n
    if ta == "r" and tb == "s":
        return "s", (kb - ka) % n
    if ta == "s" and tb == "r":
        return "s", (ka + kb) % n
    if ta == "s" and tb == "s":
        return "r", (kb - ka) % n

    raise ValueError("Invalid dihedral element")


def build_dihedral_transition(n: int, generators: list[Element], lazy_prob: float = 0.5) -> np.ndarray:
    elems = dihedral_elements(n)
    idx = {g: i for i, g in enumerate(elems)}
    m = len(elems)
    transition = np.zeros((m, m), dtype=float)

    for x in elems:
        i = idx[x]
        transition[i, i] += lazy_prob
        move_prob = (1.0 - lazy_prob) / len(generators)
        for s in generators:
            y = dihedral_multiply(s, x, n)
            j = idx[y]
            transition[i, j] += move_prob

    return transition


def run_dihedral_experiments(
    out_dir: Path,
    max_steps: int = 120,
) -> list[dict[str, object]]:
    figures_dir = out_dir / "figures"
    ensure_dir(figures_dir)

    configs = [
        {"group": "D_10", "n": 10, "generators": [("r", 1), ("r", 9), ("s", 0)]},
        {"group": "D_20", "n": 20, "generators": [("r", 1), ("r", 19), ("s", 0)]},
        {"group": "D_20_alt", "n": 20, "generators": [("r", 1), ("s", 0)]},
    ]

    rows: list[dict[str, object]] = []

    for cfg in configs:
        group_name = str(cfg["group"])
        n = int(cfg["n"])
        generators = list(cfg["generators"])

        transition = build_dihedral_transition(n=n, generators=generators, lazy_prob=0.5)
        steps, tvs = mixing_curve(transition=transition, start_idx=0, max_steps=max_steps)
        gap = markov_spectral_gap(transition)
        t_mix = mixing_time(transition=transition, start_idx=0, max_steps=max_steps, threshold=0.01)

        slug = f"dihedral_{group_name}"

        save_curve_plot(
            steps=steps,
            values=tvs,
            out_path=figures_dir / f"{slug}_tv.png",
            title=f"{group_name} | generators={generators}",
            ylabel="TV distance",
        )
        save_probability_heatmap(
            transition=transition,
            start_idx=0,
            max_steps=max_steps,
            out_path=figures_dir / f"{slug}_heatmap.png",
            title=f"{group_name} heatmap",
        )

        rows.append(
            {
                "family": "dihedral",
                "group": group_name,
                "size": 2 * n,
                "generators": str(generators),
                "spectral_gap": gap,
                "mixing_time_eps_0.01": t_mix,
            }
        )

    return rows
