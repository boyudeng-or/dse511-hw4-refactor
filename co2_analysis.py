"""Clean the Our World in Data CO2 table and compare five high-income economies.

Refactored version of the DSE 511 HW2 notebook (``original/eda_original.ipynb``).
Pipeline: load -> clean whitespace -> drop sparse rows -> focus-country subset
-> latest-year snapshot -> printed summary and saved figures.

Usage
-----
    python co2_analysis.py                          # CSV next to this file
    python co2_analysis.py --data path/to/owid-co2-data.csv --show

The data path can also be supplied through the ``CO2_DATA_PATH`` environment
variable (used by the tests).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_URL = "https://owid-public.owid.io/data/co2/owid-co2-data.csv"
DATA_PATH_ENV_VAR = "CO2_DATA_PATH"

ANALYSIS_COLUMNS = ("country", "year", "co2", "co2_per_capita", "gdp")
NUMERIC_COLUMNS = ("year", "co2", "co2_per_capita", "gdp")

# (column, axis label) for each time-series figure; the label is reused in the title.
TIME_SERIES_PLOTS = (
    ("co2", "CO2 emissions"),
    ("co2_per_capita", "CO2 per capita"),
)


def default_data_path() -> Path:
    """Return the CSV location from ``CO2_DATA_PATH``, else the script directory."""
    env_value = os.environ.get(DATA_PATH_ENV_VAR)
    return Path(env_value) if env_value else SCRIPT_DIR / "owid-co2-data.csv"


@dataclass(frozen=True)
class AnalysisConfig:
    """All tunable settings for one run, kept in one place instead of scattered literals.

    Attributes
    ----------
    data_path : Path
        Location of ``owid-co2-data.csv``.
    output_dir : Path
        Folder where figures are written.
    max_missing_frac : float
        Rows with a strictly larger fraction of missing cells are dropped.
    focus_countries : tuple of str
        Countries compared in the EDA, in legend order.
    start_year : int
        First year (inclusive) kept in the focus subset.
    """

    data_path: Path = field(default_factory=default_data_path)
    output_dir: Path = SCRIPT_DIR / "figures"
    # The table is historical, so imputing (e.g. with a column mean) would invent
    # emissions that were never measured; nearly empty rows are dropped instead.
    max_missing_frac: float = 0.90
    # Similar high-income economies with fairly complete post-1990 records,
    # few enough that the line plots stay readable.
    focus_countries: tuple[str, ...] = (
        "United States",
        "Japan",
        "Germany",
        "South Korea",
        "United Kingdom",
    )
    start_year: int = 1990


@dataclass
class AnalysisResults:
    """Intermediate and final tables produced by :func:`run_analysis`."""

    raw_shape: tuple[int, int]
    blank_cell_count: int
    cleaned: pd.DataFrame
    focus: pd.DataFrame
    snapshot: pd.DataFrame


# ---------------------------------------------------------------------------
# Data loading and cleaning
# ---------------------------------------------------------------------------

def load_data(path: Path | str) -> pd.DataFrame:
    """Read the raw OWID CO2 CSV.

    Parameters
    ----------
    path : Path or str
        Location of ``owid-co2-data.csv``.

    Returns
    -------
    pandas.DataFrame
        The table as stored on disk, including the space-padded column names
        and cell values that :func:`clean_whitespace` removes.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist. The CSV (~100 MB) is not bundled with the
        code; download it from ``DATA_URL``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Download it from {DATA_URL} and pass --data, or set {DATA_PATH_ENV_VAR}."
        )
    return pd.read_csv(path)


def _text_columns(df: pd.DataFrame) -> pd.Index:
    # pandas >= 3 reads text as the "str" dtype rather than object, so both must be selected.
    return df.select_dtypes(include=["object", "string"]).columns


def count_blank_cells(df: pd.DataFrame) -> int:
    """Count text cells that are empty or contain only whitespace.

    Parameters
    ----------
    df : pandas.DataFrame
        Any table; only text (object/str) columns are inspected.

    Returns
    -------
    int
        Number of blank text cells. Real ``NaN`` values are not counted.
    """
    return int(sum(df[col].str.strip().eq("").sum() for col in _text_columns(df)))


def clean_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Strip padding from column names and text cells, and mark blank cells missing.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw table. Column names are assumed to be strings.

    Returns
    -------
    pandas.DataFrame
        A new table (the input is not modified) where column names and text
        values have no leading/trailing whitespace and blank text cells are
        ``NaN``. Numeric-looking text columns are left as text; conversion is
        done only for the columns the analysis needs (see
        :func:`prepare_focus_subset`).
    """
    cleaned = df.rename(columns=lambda name: name.strip())
    text_cols = _text_columns(cleaned)
    cleaned[text_cols] = cleaned[text_cols].apply(lambda col: col.str.strip())
    return cleaned.replace("", np.nan)


def drop_sparse_rows(df: pd.DataFrame, max_missing_frac: float = 0.90) -> pd.DataFrame:
    """Remove rows whose fraction of missing cells exceeds a threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        Table in which blanks are already ``NaN`` (run :func:`clean_whitespace` first).
    max_missing_frac : float, default 0.90
        Rows with a missing fraction strictly greater than this are dropped;
        a row exactly at the threshold is kept.

    Returns
    -------
    pandas.DataFrame
        The remaining rows with their original index labels.

    Raises
    ------
    ValueError
        If ``max_missing_frac`` is outside [0, 1].
    """
    if not 0.0 <= max_missing_frac <= 1.0:
        raise ValueError(f"max_missing_frac must be in [0, 1], got {max_missing_frac}")
    too_sparse = df.isna().mean(axis=1) > max_missing_frac
    return df.loc[~too_sparse]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def prepare_focus_subset(
    df: pd.DataFrame,
    countries: tuple[str, ...] | list[str],
    start_year: int,
) -> pd.DataFrame:
    """Select the analysis columns for the focus countries from ``start_year`` on.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned table containing every column in ``ANALYSIS_COLUMNS``.
    countries : sequence of str
        Exact country names to keep.
    start_year : int
        First year to keep (inclusive).

    Returns
    -------
    pandas.DataFrame
        Columns ``ANALYSIS_COLUMNS`` with ``NUMERIC_COLUMNS`` converted to numbers.
        Values that cannot be parsed become ``NaN`` rather than raising.

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    missing = [col for col in ANALYSIS_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    subset = df[list(ANALYSIS_COLUMNS)].copy()
    numeric = list(NUMERIC_COLUMNS)
    subset[numeric] = subset[numeric].apply(pd.to_numeric, errors="coerce")

    keep = subset["country"].isin(countries) & (subset["year"] >= start_year)
    return subset.loc[keep].copy()


def latest_year_snapshot(df: pd.DataFrame, sort_by: str = "co2") -> pd.DataFrame:
    """Return the rows from the most recent year, largest ``sort_by`` first.

    Parameters
    ----------
    df : pandas.DataFrame
        Focus subset with a numeric ``year`` column.
    sort_by : str, default "co2"
        Column used to rank countries (descending).

    Returns
    -------
    pandas.DataFrame
        One row per country present in the latest year. A country with no row
        in that year simply does not appear.

    Raises
    ------
    ValueError
        If ``df`` is empty, since there is no latest year to report.
    """
    if df.empty:
        raise ValueError("Cannot take a latest-year snapshot of an empty table.")
    latest_year = df["year"].max()
    return df[df["year"] == latest_year].sort_values(sort_by, ascending=False)


def run_analysis(config: AnalysisConfig) -> AnalysisResults:
    """Run the full data pipeline without plotting or printing.

    Parameters
    ----------
    config : AnalysisConfig
        Data location and analysis settings.

    Returns
    -------
    AnalysisResults
        Raw shape, blank-cell count, cleaned table, focus subset, and snapshot.
    """
    raw = load_data(config.data_path)
    cleaned = drop_sparse_rows(clean_whitespace(raw), config.max_missing_frac)
    focus = prepare_focus_subset(cleaned, config.focus_countries, config.start_year)
    return AnalysisResults(
        raw_shape=raw.shape,
        blank_cell_count=count_blank_cells(raw),
        cleaned=cleaned,
        focus=focus,
        snapshot=latest_year_snapshot(focus),
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def plot_metric_over_time(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    countries: tuple[str, ...] | list[str],
) -> plt.Figure:
    """Draw one line per country for ``metric`` against year.

    Parameters
    ----------
    df : pandas.DataFrame
        Focus subset with ``country``, ``year`` and ``metric`` columns.
    metric : str
        Column to plot on the y-axis.
    ylabel : str
        Axis label, also used in the title.
    countries : sequence of str
        Countries to draw, in legend order.

    Returns
    -------
    matplotlib.figure.Figure
        The figure; the caller decides whether to show or save it.
    """
    fig, ax = plt.subplots()
    for country in countries:
        rows = df[df["country"] == country]
        ax.plot(rows["year"], rows[metric], label=country)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} over time (selected countries)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_latest_snapshot(
    snapshot: pd.DataFrame,
    metric: str = "co2",
    ylabel: str = "CO2 emissions",
) -> plt.Figure:
    """Draw a bar chart of ``metric`` for each country in the snapshot year.

    Parameters
    ----------
    snapshot : pandas.DataFrame
        Output of :func:`latest_year_snapshot` (already sorted).
    metric : str, default "co2"
        Column to plot.
    ylabel : str, default "CO2 emissions"
        Axis label, also used in the title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    year = int(snapshot["year"].iloc[0])
    fig, ax = plt.subplots()
    ax.bar(snapshot["country"], snapshot[metric])
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by country ({year})")
    fig.tight_layout()
    return fig


def build_figures(results: AnalysisResults, config: AnalysisConfig) -> dict[str, plt.Figure]:
    """Create all EDA figures, keyed by the file stem used when saving."""
    figures = {
        f"{metric}_over_time": plot_metric_over_time(
            results.focus, metric, label, config.focus_countries
        )
        for metric, label in TIME_SERIES_PLOTS
    }
    figures["co2_latest_year"] = plot_latest_snapshot(results.snapshot)
    return figures


def save_figures(figures: dict[str, plt.Figure], output_dir: Path) -> list[Path]:
    """Save each figure as ``<output_dir>/<name>.png`` and return the paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fig in figures.items():
        path = output_dir / f"{name}.png"
        fig.savefig(path, dpi=150)
        paths.append(path)
    return paths


def print_summary(results: AnalysisResults) -> None:
    """Print the data-quality checks and the latest-year comparison table."""
    print(f"Raw data shape: {results.raw_shape}")
    print(f"Blank (whitespace-only) cells treated as missing: {results.blank_cell_count}")
    print(f"Shape after dropping sparse rows: {results.cleaned.shape}")
    print(f"\nFocus subset: {len(results.focus)} rows")
    print(results.focus["country"].value_counts().to_string())
    print("\nLatest-year comparison:")
    print(results.snapshot[["country", "year", "co2", "co2_per_capita"]].to_string())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=None,
                        help="path to owid-co2-data.csv (default: CO2_DATA_PATH or script folder)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="folder for figures (default: ./figures next to this script)")
    parser.add_argument("--show", action="store_true",
                        help="also display figures interactively")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> AnalysisResults:
    """Command-line entry point: run the analysis, print the summary, save figures."""
    args = parse_args(argv)
    overrides = {}
    if args.data is not None:
        overrides["data_path"] = args.data
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    config = AnalysisConfig(**overrides)

    results = run_analysis(config)
    print_summary(results)

    figures = build_figures(results, config)
    for path in save_figures(figures, config.output_dir):
        print(f"Saved {path}")
    if args.show:
        plt.show()
    plt.close("all")
    return results


if __name__ == "__main__":
    main()
