from __future__ import annotations

from pathlib import Path

from common import ensure_dir, write_json
from cyclic import run_cyclic_experiments
from dihedral import run_dihedral_experiments
from mathlab.repro import env_info_dict
from symmetric import run_symmetric_experiments


def main() -> None:
    here = Path(__file__).resolve().parent
    out_dir = here / "artifacts"
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"

    ensure_dir(out_dir)
    ensure_dir(figures_dir)
    ensure_dir(tables_dir)

    metadata = {
        "experiment": here.name,
        "env": env_info_dict(),
    }

    rows = []
    rows.extend(run_cyclic_experiments(out_dir=out_dir, max_steps=120))
    rows.extend(run_dihedral_experiments(out_dir=out_dir, max_steps=120))
    rows.extend(run_symmetric_experiments(out_dir=out_dir, max_steps=80))

    summary = {
        "num_experiments": len(rows),
        "families": sorted({row["family"] for row in rows}),
        "groups": [row["group"] for row in rows],
    }
    metadata["summary"] = summary

    print(f"Wrote: {out_dir / 'metadata.json'}")
    print(f"Wrote: {tables_dir / 'summary.csv'}")
    print(f"Wrote: {out_dir / 'summary.json'}")
    print(f"Figures dir: {figures_dir}")

    write_json(out_dir / "metadata.json", metadata)


if __name__ == "__main__":
    main()
