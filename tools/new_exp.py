from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
TEMPLATE = EXPERIMENTS / "_template"


@dataclass(frozen=True)
class ExpSpec:
    slug: str
    title: str
    folder: str


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    if not s:
        raise ValueError("empty slug after slugify")
    return s


def make_spec(title: str) -> ExpSpec:
    slug = slugify(title)
    folder = f"{date.today().isoformat()}-{slug}"
    return ExpSpec(slug=slug, title=title.strip(), folder=folder)


def copy_tree(src: Path, dst: Path) -> None:
    for p in src.rglob("*"):
        rel = p.relative_to(src)
        target = dst / rel
        if p.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(p.read_bytes())


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("title", help="Experiment title, used to generate folder slug.")
    args = parser.parse_args()

    spec = make_spec(args.title)
    dst = EXPERIMENTS / spec.folder
    if dst.exists():
        raise SystemExit(f"Experiment already exists: {dst}")

    copy_tree(TEMPLATE, dst)

    readme = dst / "README.md"
    text = readme.read_text(encoding="utf-8")
    text = text.replace("{{TITLE}}", spec.title)
    text = text.replace("{{FOLDER}}", spec.folder)
    readme.write_text(text, encoding="utf-8")

    print(f"Created: {dst}")


if __name__ == "__main__":
    main()
