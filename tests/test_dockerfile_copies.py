"""Every first-party package a service imports must be COPYed into its image.

Both service images have shipped broken this way:

  * backend/Dockerfile copied backend/ and ai/ but not configs/, while
    backend/pipeline imports `from configs import registry` -- the image built
    fine and failed on the first processing job.
  * nexus/Dockerfile copied only nexus/, while nexus/sports/game_plan.py
    imports ai.opponent_intelligence.* -- the container exited at startup with
    ModuleNotFoundError: No module named 'ai'.

Neither was caught by the test suite, because on a developer machine every
package is importable from the repo root. This test reads the Dockerfiles and
the imports, so the next missing COPY fails here instead of in a container.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIRST_PARTY = {"ai", "backend", "nexus", "configs", "training", "scripts"}

SERVICES = {
    "backend": ("backend/Dockerfile", "backend"),
    "nexus": ("nexus/Dockerfile", "nexus"),
}


def _copied_paths(dockerfile: Path) -> set[str]:
    """Top-level repo directories the Dockerfile COPYs from the build context."""
    copied: set[str] = set()
    text = dockerfile.read_text(encoding="utf-8")
    text = re.sub(r"\\\s*\n", " ", text)  # join line continuations
    for line in text.splitlines():
        line = line.strip()
        if not line.upper().startswith("COPY "):
            continue
        parts = line.split()[1:]
        parts = [p for p in parts if not p.startswith("--")]  # skip --from=build etc.
        for source in parts[:-1]:  # last token is the destination
            copied.add(source.split("/")[0].lstrip("./"))
    return copied


def _first_party_imports(package: str) -> dict[str, str]:
    """top-level first-party package -> the file that imports it."""
    found: dict[str, str] = {}
    for path in sorted((REPO_ROOT / package).rglob("*.py")):
        if "tests" in path.parts or "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - not expected
            continue
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules = [node.module]
            for module in modules:
                top = module.split(".")[0]
                if top in FIRST_PARTY and top != package:
                    found.setdefault(top, str(path.relative_to(REPO_ROOT)))
    return found


@pytest.mark.parametrize("service", sorted(SERVICES))
def test_dockerfile_copies_every_imported_first_party_package(service):
    dockerfile_rel, package = SERVICES[service]
    dockerfile = REPO_ROOT / dockerfile_rel
    copied = _copied_paths(dockerfile)
    assert package in copied, f"{dockerfile_rel} does not COPY its own package {package}/"

    missing = {
        needed: importer
        for needed, importer in _first_party_imports(package).items()
        if needed not in copied
    }
    assert not missing, (
        f"{dockerfile_rel} is missing COPY for: "
        + ", ".join(f"{pkg}/ (imported by {src})" for pkg, src in sorted(missing.items()))
    )


@pytest.mark.parametrize("service", sorted(SERVICES))
def test_dockerfile_runs_as_non_root(service):
    text = (REPO_ROOT / SERVICES[service][0]).read_text(encoding="utf-8")
    assert re.search(r"^USER\s+(?!root)\S+", text, flags=re.M), "image would run as root"


@pytest.mark.parametrize("service", sorted(SERVICES))
def test_dockerfile_declares_a_healthcheck(service):
    text = (REPO_ROOT / SERVICES[service][0]).read_text(encoding="utf-8")
    assert "HEALTHCHECK" in text


def test_dockerignore_excludes_the_heavy_directories():
    """Without these the build context is >1 GB (venv, weights, datasets)."""
    ignored = {
        line.strip().rstrip("/")
        for line in (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    for entry in ("venv", "datasets", "models", "storage", "samples", "runs", ".git"):
        assert entry in ignored, f".dockerignore should exclude {entry}/"
