from __future__ import annotations

import json
from pathlib import Path

from mathlab.repro import env_info_dict


def main() -> None:
    here = Path(__file__).resolve().parent
    out_dir = here / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "experiment": here.name,
        "env": env_info_dict(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote: {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
