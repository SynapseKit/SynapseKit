from __future__ import annotations

import os
import platform
import random
import subprocess
import time
from pathlib import Path

import pytest

try:
    import numpy as np
except Exception:  # pragma: no cover - optional
    np = None

try:
    import psutil
except Exception:  # pragma: no cover - optional
    psutil = None

ROOT = Path(__file__).resolve().parents[1]
COOLDOWN_SECONDS = 0.1


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def pytest_sessionstart(session):
    os.environ.setdefault("PYTHONHASHSEED", "0")
    random.seed(0)
    if np is not None:
        np.random.seed(0)
    root = Path(__file__).resolve().parents[1]
    (root / ".benchmarks").mkdir(parents=True, exist_ok=True)


@pytest.fixture(autouse=True)
def _cooldown_between_benchmarks():
    yield
    time.sleep(COOLDOWN_SECONDS)


def pytest_benchmark_update_machine_info(config, machine_info):
    machine_info["git_sha"] = _git_sha()
    machine_info["python"] = platform.python_version()
    machine_info["os"] = platform.platform()
    machine_info["cpu"] = platform.processor()
    if psutil:
        machine_info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
