# reVRt Coding Agent Onboarding Guide

## 1. Project Summary
reVRt models and optimizes electric transmission routing using least-cost
path analysis.
- Python package `revrt` orchestrates layer preparation, routing runs, and
  CLI tooling (`reVRt`).
- Rust crate `crates/revrt` implements the high-performance routing core,
  exposed to Python through PyO3.
- Rust crate `crates/cli` ships a native CLI that mirrors the Python entry
  points and offers benchmarking harnesses.

Primary outcomes: reproducible routing layers, cost surfaces, and
benchmarkable transmission paths suitable for reV/reVX workflows or
standalone analysis.

## 2. Tech Stack Overview
- Python ≥ 3.11 for orchestration, layer management, and CLI surfaces.
- Rust 2024 edition (workspace toolchain ≥ 1.87) for routing algorithms and
  PyO3 bindings built with Maturin.
- Environment management: **Pixi** with feature-based environments (`dev`,
  `test`, `doc`, `build`). Prefer Pixi tasks over bare `pip` or `cargo`.
- Packaging: `pyproject.toml` drives both Python metadata and Pixi
  environments; the PyO3 module is packaged as `revrt._rust`.
- Linting/formatting: Ruff (line length 79, docstrings 72) with numpy-style
  docstring enforcement.
- Testing: Pytest (unit + integration) with coverage gate 95%; Rust unit tests
  via Cargo plus Criterion benchmarks.
- Docs: Sphinx + pydata theme (`docs/`), built with Pixi tasks and failing on
  warnings.

## 3. High-Level Project Structure
```
revrt/                      Core Python package (CLI, costs, models, utilities)
  costs/                    Layer builders and friction/barrier assembly
  models/                   Pydantic config and cost layer schemas
  spatial_characterization/ Derived spatial analytics
  utilities/                Shared IO, raster, logging, and helpers
crates/                     Rust code
  revrt/                    Rust routing library + PyO3 extension
  cli/                      Rust CLI wrapper and benches
docs/                       Sphinx sources and dev guidelines
tests/python/unit           Focused unit coverage (>=95%)
tests/python/integration    Scenario-level regression tests
support/                    Helper scripts (e.g., Zarr conversion)
```

## 4. Environment & Commands Cheat Sheet
Always use Pixi to ensure aligned dependencies:
```
# Enter dev shell (Python + Rust toolchain)
pixi shell -e dev

# Linting and formatting
pixi run -e dev lint
pixi run -e dev format

# Python tests (95% coverage gate)
pixi run -e dev tests-u     # unit
pixi run -e dev tests-i     # integration
pixi run -e dev tests       # full suite

# Focused unit modules (useful to check a particular feature)
pixi run -e dev pytest tests/python/unit/... -rapP -vv

# Rust tests & benches
pixi run -e dev tests-r
pixi run -e dev cargo test --locked --workspace --no-fail-fast

# Build documentation (fails on warnings)
pixi run -e dev python-docs

# Build wheels / extension
pixi run -e build build-wheel

# CLI smoke test
pixi run -e dev reVRt --help
```
To add a new dependency, use `pixi add --pipy <package>`;
this updates `pyproject.toml` and the lockfile atomically.
You will also have to run a subsequent command `pixi add <package>`
to add the dependency to the conda environment.
Use `pixi add --feature dev <package>` to add a dependency
that is only used for development (tests, linting, docs, etc.).

## 5. Coding Guidelines (Python)
- Follow `docs/source/dev/README.rst` for style: maintain numpy-style
  docstrings, avoid type hints unless pre-existing, and keep module-level
  imports in the documented order (`numpy as np`, `xarray as xr`, etc.).
- Respect Ruff defaults (line length 79; docstrings 72). Run
  `pixi run -e dev ruff check .` and `pixi run -e dev ruff format .` before
  opening a PR.
- Favor descriptive names instead of extra comments; only add comments for
  non-obvious behavior (e.g., tricky array ops or concurrency concerns).
  If some functionality needs further explanation, add this to the
  class/function/method docstring under the "Notes" section.
- Use absolute imports under `revrt.`.
- Surface warnings/errors through `revrt.warn` and `revrt.exceptions` (e.g.,
  raise `revrtValueError` and emit `revrtWarning`) to ensure logging hooks fire.
- When touching PyO3 bindings, update both Python shims in `revrt/_rust.py` (if
  needed) and the Rust exports to keep signatures aligned.

## 6. Docstring Guidelines (Python)
- Use numpy-style docstrings; first line must omit a trailing period.
- Keep docstring lines ≤72 characters and avoid short summaries on
  `__init__` methods.
- Document parameters in the method/function docstring, not the class-level
  docstring; protected helpers (`_name`) should have single-sentence docs.
- Maintain intersphinx references where possible (see dev guide for mappings).

## 7. Coding Guidelines (Rust)
- Update shared dependencies in the root `Cargo.toml`; keep workspace
  versions consistent with Python packaging (see `workspace.package`).
- Run formatting and linting via Pixi:
  `pixi run -e dev cargo fmt --all --check` and
  `pixi run -e dev cargo clippy --all-targets -- -D warnings` before commits.
- Expose new functionality via the PyO3 module (`crates/revrt/src/lib.rs`) and
  ensure Python bindings validate inputs that Rust assumes.

## 8. Testing Strategy
- Place unit tests near related modules (`tests/python/unit/**`); integration
  scenarios belong under `tests/python/integration`.
- The Pytest tasks enforce `--cov-fail-under=95`; add or update tests until the
  threshold passes. Coverage reports land in `htmlcov/` and `coverage.xml`.
- Rust tests/benches live within each crate (`cargo test`, `cargo bench` via
  Pixi). Keep tests deterministic and avoid long-running benchmarks in CI.
- Include regression data under `tests/data/` when necessary; prefer fixtures in
  `tests/conftest.py` for shared setup. Keep all test data small (individual files <1MB).
  The data doesn't need to fully reproduce realistic analysis cases - it just
  needs to include characteristics of a realistic case.
- Pytest options for parallel execution (`-n auto`) are supported; prefer
  `pixi run -e dev pytest -n auto` for heavier suites.

## 9. Logging, Errors, and Warnings
- Do not log-and-raise manually; custom exceptions/warnings already emit log
  records.
- Prefer `revrt.utilities.log_mem` for memory-sensitive workflows to keep log
  output consistent.
- CLI commands should rely on the logging configuration provided in
  `revrt._cli.configure_logging` to avoid duplicate handlers.

## 10. Common Pitfalls & Gotchas
- The PyO3 module must be rebuilt if Rust code changes; run a Pixi task (tests
  or wheel build) to regenerate the extension before importing Python modules.
- Coverage gates are strict—unreachable code paths or broad exception handlers
  often reduce coverage; refactor or mark as defensive with justification.
- Large raster operations can exhaust memory; use Dask abstractions already in
  `revrt.costs` and respect chunk metadata stored in layer files.
- Versioning follows semantic triples per component (`vX.Y.Z` Python package,
  `rX.Y.Z` Rust core, `cX.Y.Z` CLI). Coordinate bumps rather than editing in
  isolation.

## 11. Implementing a Sample Feature (Reference Flow)
Example: Add a new CLI subcommand that dumps routing metadata.
1. Extend `revrt/_cli.py` with a new `click` command that orchestrates the
   behavior.
2. Add supporting logic under an appropriate module (e.g.,
   `revrt/utilities/metadata.py`).
3. Expose the functionality in the Rust crate if performance-critical, and
   update PyO3 bindings accordingly.
4. Add unit tests (`tests/python/unit/_cli/`) plus integration coverage if the
   command touches workflows.
5. Run `pixi run -e dev tests-u` (and others as needed) before raising a PR.

## 12. CI/Release Lifecycle
- CI runs Ruff, Pytest (unit + integration), Cargo checks/tests, Criterion
  benchmarks, and docs. Fix lint/test issues locally before pushing.
- Docs must build without warnings (`pixi run -e dev python-docs`).
- Publishing wheels uses `pixi run -e build build-wheel`; coordinate version
  bumps across Python and Rust artifacts per the dev guide.

## 13. Contribution & Style References
- Consult `docs/source/dev/README.rst` and `docs/source/dev/development_notes.md`
  for extended guidelines.
- Keep functions cohesive; prefer helper utilities in `revrt.utilities` over
  expanding class responsibilities.
- Follow documented import conventions and reuse existing enums/constants from
  `revrt.constants` when possible.

## 14. Quick Triage Guide for the Agent
- Cost layer logic: `revrt/costs/` (Zarr I/O, raster assembly, masks).
- Routing outputs and models: `revrt/models/` and `revrt/costs/config/`.
- Spatial analytics: `revrt/spatial_characterization/`.
- CLI behavior: `revrt/_cli.py` and `crates/cli/src/main.rs`.
- Exceptions/warnings: `revrt/exceptions.py`, `revrt/warn.py`.

## 15. Security & Secrets
- Do not commit credentials. External services (e.g., data sources) should be
  configured via environment variables or Pixi-managed secrets.
- Review the CI workflows before altering release automation; they assume
  reproducible Pixi environments and locked dependencies.
