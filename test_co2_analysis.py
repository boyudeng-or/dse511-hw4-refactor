"""Tests for co2_analysis.py.

Run with:  pytest -v test_co2_analysis.py

Unit tests use small hand-built tables. The behavior-preservation test executes
the original HW2 notebook and needs the real CSV; it is skipped if the file is
not found (place it next to co2_analysis.py or set CO2_DATA_PATH).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import co2_analysis as ca

ORIGINAL_NOTEBOOK = Path(__file__).resolve().parent / "original" / "eda_original.ipynb"


@pytest.fixture
def padded_raw() -> pd.DataFrame:
    """Mimics the raw CSV: padded column names, padded values, whitespace-only blanks."""
    return pd.DataFrame(
        {
            "country   ": ["Japan     ", "  Germany ", "Chad      "],
            " year": [2020, 2021, 2022],
            "co2       ": ["  961.8   ", "          ", " 1.5      "],
        }
    )


def make_focus_input(rows: list[tuple]) -> pd.DataFrame:
    """Build a cleaned-style table (all text) with the columns prepare_focus_subset needs."""
    return pd.DataFrame(rows, columns=list(ca.ANALYSIS_COLUMNS))


# ---------------------------------------------------------------------------
# Normal-case tests
# ---------------------------------------------------------------------------

def test_clean_whitespace_strips_names_and_values(padded_raw):
    cleaned = ca.clean_whitespace(padded_raw)

    assert list(cleaned.columns) == ["country", "year", "co2"]
    assert cleaned["country"].tolist() == ["Japan", "Germany", "Chad"]
    assert cleaned.loc[0, "co2"] == "961.8"
    assert pd.isna(cleaned.loc[1, "co2"])


def test_clean_whitespace_does_not_modify_input(padded_raw):
    before = padded_raw.copy()
    ca.clean_whitespace(padded_raw)
    pd.testing.assert_frame_equal(padded_raw, before)


def test_count_blank_cells_ignores_real_nan(padded_raw):
    padded_raw.loc[2, "country   "] = np.nan
    assert ca.count_blank_cells(padded_raw) == 1


def test_prepare_focus_subset_filters_and_converts():
    df = make_focus_input([
        ("Japan", "1989", "1000.0", "8.0", "1e12"),   # before start year
        ("Japan", "2024", "961.9", "7.8", None),
        ("Germany", "2024", "572.3", "6.8", "4e12"),
        ("Chad", "2024", "1.5", "0.1", "1e10"),       # not a focus country
    ])

    subset = ca.prepare_focus_subset(df, ["Japan", "Germany"], start_year=1990)

    assert subset["country"].tolist() == ["Japan", "Germany"]
    assert subset["co2"].tolist() == [961.9, 572.3]
    assert pd.api.types.is_numeric_dtype(subset["year"])
    assert pd.isna(subset["gdp"].iloc[0])


def test_latest_year_snapshot_keeps_latest_year_sorted_by_co2():
    df = pd.DataFrame({
        "country": ["A", "B", "A", "B", "C"],
        "year": [2023, 2023, 2024, 2024, 2024],
        "co2": [5.0, 1.0, 2.0, 9.0, 4.0],
    })

    snapshot = ca.latest_year_snapshot(df)

    assert snapshot["country"].tolist() == ["B", "C", "A"]
    assert (snapshot["year"] == 2024).all()


# ---------------------------------------------------------------------------
# Edge-case and invalid-input tests
# ---------------------------------------------------------------------------

def test_drop_sparse_rows_threshold_is_strict():
    # 10 columns: row 0 is 10% missing, row 1 exactly 90%, row 2 100%.
    data = np.ones((3, 10))
    data[1, 1:] = np.nan
    data[2, :] = np.nan
    df = pd.DataFrame(data)

    kept = ca.drop_sparse_rows(df, max_missing_frac=0.90)

    assert kept.index.tolist() == [0, 1]


@pytest.mark.parametrize("bad_frac", [-0.1, 1.5])
def test_drop_sparse_rows_rejects_invalid_threshold(bad_frac):
    with pytest.raises(ValueError, match="max_missing_frac"):
        ca.drop_sparse_rows(pd.DataFrame({"a": [1.0]}), max_missing_frac=bad_frac)


def test_drop_sparse_rows_empty_input_returns_empty():
    empty = pd.DataFrame(columns=["country", "year"])
    assert ca.drop_sparse_rows(empty).empty


def test_prepare_focus_subset_missing_column_raises():
    df = make_focus_input([("Japan", "2024", "1.0", "1.0", "1.0")]).drop(columns="gdp")
    with pytest.raises(ValueError, match="gdp"):
        ca.prepare_focus_subset(df, ["Japan"], start_year=1990)


def test_prepare_focus_subset_unparseable_number_becomes_nan():
    df = make_focus_input([("Japan", "2024", "n/a", "7.8", "1e12")])
    subset = ca.prepare_focus_subset(df, ["Japan"], start_year=1990)
    assert pd.isna(subset["co2"].iloc[0])


def test_latest_year_snapshot_empty_raises():
    empty = pd.DataFrame(columns=["country", "year", "co2"])
    with pytest.raises(ValueError, match="empty"):
        ca.latest_year_snapshot(empty)


def test_load_data_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Download"):
        ca.load_data(tmp_path / "does_not_exist.csv")


# ---------------------------------------------------------------------------
# Behavior preservation: original notebook vs refactored pipeline
# ---------------------------------------------------------------------------

def run_original_notebook(data_path: Path, monkeypatch) -> dict:
    """Execute every code cell of the untouched HW2 notebook and return its namespace."""
    cells = json.loads(ORIGINAL_NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    source = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")

    # The notebook reads "owid-co2-data.csv" relative to the working directory.
    monkeypatch.chdir(data_path.parent)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    namespace: dict = {}
    exec(compile(source, str(ORIGINAL_NOTEBOOK), "exec"), namespace)
    plt.close("all")
    return namespace


def test_refactored_matches_original_notebook(monkeypatch):
    data_path = ca.default_data_path()
    if not data_path.is_file():
        pytest.skip(f"real data not available at {data_path}")
    if data_path.name != "owid-co2-data.csv":
        pytest.skip("the original notebook only reads a file named owid-co2-data.csv")

    original = run_original_notebook(data_path, monkeypatch)
    refactored = ca.run_analysis(ca.AnalysisConfig(data_path=data_path))

    assert refactored.cleaned.shape == original["df"].shape
    assert list(refactored.cleaned.columns) == list(original["df"].columns)
    pd.testing.assert_frame_equal(refactored.focus, original["eda"])
    pd.testing.assert_frame_equal(refactored.snapshot, original["snap"])
