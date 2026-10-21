"""The project must be self-contained and originally authored.

No code from any sibling repository under promotion/code -- not imported, not
copied, not adapted -- and no git or path dependency sources. This is a hard
requirement for the published artifact, so it is enforced rather than trusted.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE = REPO_ROOT / "di_ssl_agri_gap"

SIBLING_PACKAGES = [
    "logreg",
    "vision_datasets",
    "rbfn_head",
    "rbfn_ssl_projector",
    "simclr",
    "simsiam",
    "moco",
    "byol",
    "swav",
]


def python_sources() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py")) + sorted((REPO_ROOT / "tests").rglob("*.py"))


@pytest.mark.parametrize("sibling", SIBLING_PACKAGES)
def test_no_sibling_package_is_imported(sibling):
    pattern = re.compile(
        rf"^\s*(from\s+{sibling}\b|import\s+{sibling}\b)", re.MULTILINE
    )
    offenders = [
        path.relative_to(REPO_ROOT)
        for path in python_sources()
        if pattern.search(path.read_text())
    ]
    assert not offenders, f"{sibling} imported in {offenders}"


def test_no_git_or_path_dependency_sources():
    """A git= or path= source would make the lockfile non-reproducible."""
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    poetry = data["tool"]["poetry"]
    groups = [poetry.get("dependencies", {})]
    for group in poetry.get("group", {}).values():
        groups.append(group.get("dependencies", {}))

    for deps in groups:
        for name, spec in deps.items():
            if isinstance(spec, dict):
                assert "git" not in spec, f"{name} uses a git source"
                assert "path" not in spec, f"{name} uses a path source"
                assert "url" not in spec, f"{name} uses a url source"


def test_package_imports_without_sibling_repos_on_the_path():
    """Every module must import cleanly from public dependencies alone."""
    import importlib

    for name in (
        "enums",
        "splits",
        "taxonomy",
        "utils",
        "prepare",
        "manifest",
        "resize",
        "stage_shm",
        "stream",
    ):
        importlib.import_module(f"di_ssl_agri_gap.{name}")
