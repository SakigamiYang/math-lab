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


def build_cyclic_transition(n: int, generators: list[int], lazy_prob: float = 0.5) -> np.ndarray:
    """
    Lazy random walk on Z_n:
        P = lazy_prob * I + (1-lazy_prob)/|S| * sum_{s in S} shift by s
    """
    transition = np.zeros((n, n), dtype=float)

    for x in range(n):
        transition[x, x] += lazy_prob
        move_prob = (1.0 - lazy_prob) / len(generators)
        for g in generators:
            y = (x + g) % n
            transition[x, y] += move_prob

    return transition


def run_cyclic_experiments(
    out_dir: Path,
    max_steps: int = 120,
) -> list[dict[str, object]]:
    figures_dir = out_dir / "figures"
    ensure_dir(figures_dir)

    configs = [
        {"group": "Z_20", "n": 20, "generators": [1, -1]},
        {"group": "Z_20", "n": 20, "generators": [1, -1, 2, -2]},
        {"group": "Z_30", "n": 30, "generators": [1, -1]},
    ]

    rows: list[dict[str, object]] = []

    for cfg in configs:
        group_name = str(cfg["group"])
        n = int(cfg["n"])
        generators = list(cfg["generators"])

        transition = build_cyclic_transition(n=n, generators=generators, lazy_prob=0.5)
        steps, tvs = mixing_curve(transition=transition, start_idx=0, max_steps=max_steps)
        gap = markov_spectral_gap(transition)
        t_mix = mixing_time(transition=transition, start_idx=0, max_steps=max_steps, threshold=0.01)

        gen_label = ",".join(str(g) for g in generators)
        slug = f"cyclic_{group_name}_gens_{gen_label.replace('-', 'm').replace(',', '_')}"

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
            title=f"{group_name} heatmap | generators={generators}",
        )

        rows.append(
            {
                "family": "cyclic",
                "group": group_name,
                "size": n,
                "generators": str(generators),
                "spectral_gap": gap,
                "mixing_time_eps_0.01": t_mix,
            }
        )

    return rows
