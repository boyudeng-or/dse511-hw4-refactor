# DSE 511 HW4 — Refactoring the HW2 CO₂ Notebook

**Boyu Deng (bdeng3)** · Option A (own code) · Python

**Files:** `original/eda_original.ipynb` (unchanged HW2 notebook, identical to [`eda.ipynb` at commit 29ad270](https://github.com/boyudeng-or/dse511-hw2-eda/blob/29ad2700db98d82657905b61a6cdd39f6edbed80/eda.ipynb)), `co2_analysis.py` (refactored), `test_co2_analysis.py` (tests), `figures/` (output).
**Run:** `pip install pandas numpy matplotlib pytest`; put the [OWID CSV](https://owid-public.owid.io/data/co2/owid-co2-data.csv) next to the script as `owid-co2-data.csv` (or set `CO2_DATA_PATH`); then `python co2_analysis.py [--show]` and `pytest -v test_co2_analysis.py`.

### 1. Original code and its problems
The HW2 notebook cleaned the Our World in Data CO₂ table (space-padded names and values) and compared total and per-capita CO₂ for five high-income countries since 1990. My partner wrote the cleaning cells and I wrote the EDA cells; I refactored both. Problems:
- **Duplication:** the same `try/except` check twice, four near-identical `pd.to_numeric` lines, and two plot cells that differed only by column.
- **Names, state, literals:** `colums` next to `columns`, vague names (`work`, `snap`, `temp`), a dead counter, one global `df` mutated in place, and a hard-coded file name, threshold, country list and start year.
- **Hidden bugs:** `is_empty()` returned `None` for non-blank strings. Under pandas ≥ 3, text columns use the `str` dtype rather than `object`, so the "strip values" step silently did nothing; results were right only because the EDA cell stripped `country` again.

### 2. Changes
- I split the work into single-purpose functions. `load_data`, `clean_whitespace`, `count_blank_cells`, `drop_sparse_rows`, `prepare_focus_subset` and `latest_year_snapshot` are chained by `run_analysis`, which does no plotting or printing. The plotting, saving and printing functions form a separate output layer behind `main`.
- I removed the duplication: one plotting function is looped over `(column, label)` pairs, and a single vectorised conversion replaces the four `pd.to_numeric` lines.
- Settings live in a frozen `AnalysisConfig`, and the data path comes from a CLI flag or environment variable. Functions return new DataFrames instead of mutating a global, and they raise clear errors for a missing file, missing columns, a threshold outside [0, 1], or an empty table.
- Every function has a NumPy-style docstring (purpose, parameters, return value, assumptions). Comments explain only *why*, for example why sparse rows are dropped rather than imputed.

### 3. Why it is better
Each step can be read, tested and reused on its own. Changing the threshold or the countries is a one-line config change. Separating the analysis from plotting makes the numbers testable, and removing global state ends the "run cells in the right order" problem.

### 4. Verification
`test_refactored_matches_original_notebook` **executes the untouched original notebook** on the real data and compares it with the refactored pipeline. The cleaned shape (43,019 × 79) and column names match, and `assert_frame_equal` passes on the 175-row focus subset and the 2024 snapshot (US 4904 > Japan 962 > South Korea 584 > Germany 572 > UK 313 Mt). Thirteen unit tests cover normal cases (stripping, conversion, filtering, ranking) and edge cases: a row at exactly 90 % missing is kept while an empty row is dropped, an invalid threshold, empty input, a missing column, an unparseable number, and a missing file. All 14 tests pass.

### 5. Value for future me and collaborators
A collaborator can run one command, read the docstrings for each step's assumptions, and change settings without touching the logic. The tests act as a safety net; the pandas 3 bug shows that library upgrades can silently change behavior.

**Limitation:** most numeric columns stay as text after cleaning, because the original only converted the four columns it used. I kept that scope rather than add features. The figures are checked by eye, not by automated tests.
