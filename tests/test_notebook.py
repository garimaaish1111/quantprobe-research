"""The Colab notebook must be valid before it is handed to anyone.

A notebook is shipped as JSON, so a broken cell is not caught by anything -
not the linter, not the test suite, not import. It surfaces as a SyntaxError
in front of the person running it, after they have already spent minutes on
setup cells. That happened once (a literal newline inside a string literal in
the repo-upload cell); this makes it not happen twice.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))


def python_cells(nb: dict):
    """Code cells that are Python, skipping IPython shell/magic cells."""
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = cell["source"]
        if isinstance(src, list):
            src = "".join(src)
        stripped = src.lstrip()
        if stripped.startswith("!") or stripped.startswith("%"):
            continue
        yield i, src


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_json(path):
    json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_every_python_cell_compiles(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    for i, src in python_cells(nb):
        try:
            compile(src, f"{path.name}:cell{i}", "exec")
        except SyntaxError as exc:
            errors.append(f"cell {i}: line {exc.lineno}: {exc.msg}")
    assert not errors, "syntax errors in notebook cells:\n  " + "\n  ".join(errors)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_hardcoded_credential(path):
    """A saved notebook containing a live token is a leaked credential.

    Matches token SHAPE, not the bare prefix - the prefix appears legitimately
    in prose and in the getpass prompt text.
    """
    import re

    patterns = {
        "huggingface": re.compile(r"hf_[A-Za-z0-9]{20,}"),
        "openai": re.compile(r"sk-[A-Za-z0-9]{20,}"),
        "github": re.compile(r"ghp_[A-Za-z0-9]{30,}"),
        "aws": re.compile(r"AKIA[A-Z0-9]{16}"),
    }
    text = path.read_text(encoding="utf-8")
    found = {name: pat.findall(text) for name, pat in patterns.items()}
    found = {k: v for k, v in found.items() if v}
    assert not found, (
        f"{path.name} contains what looks like a live credential: "
        f"{ {k: [s[:8] + '...' for s in v] for k, v in found.items()} }"
    )


def test_notebooks_exist():
    assert NOTEBOOKS, "no notebooks found - the Colab run has nothing to run"
