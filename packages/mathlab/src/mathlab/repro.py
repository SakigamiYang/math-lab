from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EnvInfo:
    python: str
    executable: str
    platform: str
    machine: str
    processor: str
    git_commit: str | None


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def collect_env_info() -> EnvInfo:
    return EnvInfo(
        python=sys.version.replace("\n", " "),
        executable=sys.executable,
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
        git_commit=_git_commit(),
    )


def env_info_dict() -> dict:
    return asdict(collect_env_info())
