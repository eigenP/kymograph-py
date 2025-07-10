# Repository Guidelines

## Testing
- Use `pytest` for running tests located in the `tests/` folder.
- CI runs `pytest` after installing the package with `pip install .`. Ensure tests pass locally using the same command.

## Dependencies
- Development requirements are listed in `requirements-dev.txt` which installs the package in editable mode and includes `numpy` and `pytest`.
- The package itself depends on `numpy`, `pandas`, `tqdm`, `matplotlib`, `scikit-image`, and `scipy` as declared in `setup.py`.

## Python Versions
- GitHub Actions tests the code on Python 3.9, 3.11, 3.12 and 3.13.
- Keep the code compatible with these versions.

## Packaging
- `setup.py` is used for packaging via `setuptools`. Update it if new dependencies are added.

## Contributions
- Provide tests for new features or bug fixes.
- No specific linting configuration is provided; follow standard Python style.
