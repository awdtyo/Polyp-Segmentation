"""Test that the command-line flags used in notebooks/colab_eval.ipynb match the real argparse parsers.

Every flag the notebook passes to ``src.evaluate``, ``scripts/prepare_datasets.py``,
``scripts/build_cache.py`` or ``scripts/aggregate_results.py`` must exist in the
corresponding parser. The notebook builds commands in code cells; this test
extracts the ``--flag`` tokens from those cells and checks them against the
parsers of the scripts used in each cell.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "colab_eval.ipynb"

sys.path.insert(0, str(REPO_ROOT))

from src.evaluate import parse_args as eval_parse_args  # noqa: E402
from scripts.aggregate_results import parse_args as agg_parse_args  # noqa: E402
from scripts.build_cache import parse_args as build_cache_parse_args  # noqa: E402
from scripts.prepare_datasets import parse_args as prep_parse_args  # noqa: E402


def _option_strings(parse_fn, argv=None) -> set[str]:
    """Return the set of long-option names (dest form) accepted by a parser.

    Temporarily replaces ``ArgumentParser.parse_args`` so the parser object is
    returned instead of a parsed namespace (some parsers require arguments).
    """
    import argparse

    original = argparse.ArgumentParser.parse_args

    def _capture(self, args=None, namespace=None):
        return self

    argparse.ArgumentParser.parse_args = _capture
    try:
        parser = parse_fn(argv=argv or [])
    finally:
        argparse.ArgumentParser.parse_args = original

    flags: set[str] = set()
    for action in parser._actions:
        for opt in action.option_strings:
            if opt.startswith("--"):
                flags.add(opt[2:].replace("-", "_"))
    return flags


EVAL_FLAGS = _option_strings(eval_parse_args, ["--run_dir", "/tmp/x"])
PREP_FLAGS = _option_strings(prep_parse_args)
BUILD_CACHE_FLAGS = _option_strings(build_cache_parse_args)
AGG_FLAGS = _option_strings(agg_parse_args, ["--runs-dir", "/tmp/x"])

# Cell index (0-based, code cells only) -> set of flags allowed in that cell.
# Code cells are at notebook indices 1, 3, 4, 6, 8, 10.
CODE_CELL_INDICES = [1, 3, 4, 6, 8, 10]

CELL_PARSERS: dict[int, set[str]] = {
    6: PREP_FLAGS | BUILD_CACHE_FLAGS,  # data staging: build_cache + prepare_datasets
    8: EVAL_FLAGS,                       # evaluate: src.evaluate
    10: AGG_FLAGS,                       # aggregate: aggregate_results.py
}


def _notebook_code_cells():
    with open(NOTEBOOK, encoding="utf-8") as f:
        nb = json.load(f)
    return [
        (index, "".join(cell["source"]))
        for index, cell in enumerate(nb["cells"])
        if cell["cell_type"] == "code"
    ]


def _flags_in_source(source: str) -> list[str]:
    """Extract long-option names from command-building code in a cell source.

    Only flags that appear as ``--name`` tokens are returned; git flags and
    other CLI tools are excluded by only checking cells that build Python
    script commands (see CELL_PARSERS).
    """
    return [
        m.group(1).replace("-", "_")
        for m in re.finditer(r"--([A-Za-z0-9][A-Za-z0-9_-]*)", source)
    ]


@pytest.mark.parametrize("cell_index,source", _notebook_code_cells())
def test_cell_flags_match_parsers(cell_index, source):
    if cell_index not in CELL_PARSERS:
        return  # cells that do not build Python-script commands
    allowed = CELL_PARSERS[cell_index]
    for flag in _flags_in_source(source):
        assert flag in allowed, (
            f"notebook code cell {cell_index}: --{flag.replace('_', '-')} is not a "
            f"valid argument for the scripts used in that cell"
        )


def test_eval_parser_has_expected_flags():
    """Sanity check: the evaluate parser really has the flags the notebook uses."""
    expected = {
        "run_dir", "split", "checkpoint", "dataset", "tta_flip",
        "config", "results_csv", "force",
    }
    assert expected <= EVAL_FLAGS, f"missing evaluate flags: {expected - EVAL_FLAGS}"


def test_prep_parser_has_expected_flags():
    expected = {"dataset", "config", "size", "mask_threshold", "data_root",
                "out_root", "limit", "force", "quiet"}
    assert expected <= PREP_FLAGS, f"missing prepare_datasets flags: {expected - PREP_FLAGS}"


def test_agg_parser_has_expected_flags():
    expected = {"runs_dir", "out_dir", "compare", "dataset", "split",
                "tta", "bootstrap", "bootstrap_seed"}
    assert expected <= AGG_FLAGS, f"missing aggregate_results flags: {expected - AGG_FLAGS}"


def test_build_cache_parser_has_expected_flags():
    expected = {"config", "root", "size", "out", "limit"}
    assert expected <= BUILD_CACHE_FLAGS, f"missing build_cache flags: {expected - BUILD_CACHE_FLAGS}"