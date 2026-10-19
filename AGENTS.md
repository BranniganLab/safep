# Repository Guidelines

## Project Structure & Module Organization

Core package code lives in `safep/`. Parsing and file I/O are implemented in
`AFEP_parse.py` and `fileIO.py`; analysis utilities are in `processing.py` and
`estimators.py`; plotting and thermodynamic integration are in `plotting.py`
and `TI.py`. Example workflows are in `Sample_Notebooks/`, with input data in
`Sample_Notebooks/Sample_Data/`. Sphinx documentation is under `docs/`.
Place new tests in `safep/tests/` and reusable test fixtures in its `data/`
directory.

## Build, Test, and Development Commands

Use Python 3.10 or newer in a virtual environment. Install the package and
runtime dependencies with:

```bash
python -m pip install -e .
```

Run the test suite, when tests are present, with:

```bash
python -m pytest
```

Build the documentation from `docs/` with `make html` (requires the packages
listed in `docs/requirements.txt`). Build a distributable package with
`python -m build`.

## Coding Style & Naming Conventions

Use four-space indentation and follow PEP 8. Use `snake_case` for functions,
variables, and modules; `CapWords` for classes; and descriptive names for
dataframes and energy-related quantities. Add type annotations to new public
functions where practical, preserve existing pandas/numpy idioms, and keep
plotting code separate from numerical analysis. No formatter or linter is
configured, so keep changes readable and run focused tests after edits.

## Testing Guidelines

Use `pytest` and name test files `test_*.py` with test functions named
`test_<behavior>`. Cover parsers, estimators, and numerical edge cases with
small deterministic fixtures; use files in `safep/tests/data/` for parser
inputs. Run the full suite before submitting changes and run targeted tests
while developing, for example `python -m pytest safep/tests/test_estimators.py`.

## Commit & Pull Request Guidelines

Recent commits use short, imperative or action-oriented subjects such as
`Fix parameter names` and `Update README.md`. Keep commits focused and explain
user-visible or scientific changes in the body when needed. Pull requests
should describe the problem, summarize the solution, identify affected
modules or notebooks, link related issues, and include test results. Add plots
or notebook output when a change affects analysis or visualization behavior.

## Scientific and Data Considerations

Preserve units, temperature handling, and estimator conventions when changing
numerical code. Do not commit generated build artifacts, caches, credentials,
or large simulation outputs; add a small reproducible fixture instead.
