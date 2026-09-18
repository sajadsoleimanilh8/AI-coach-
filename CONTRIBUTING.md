# Contributing

## Setup

Follow RUN.md for the full Windows walkthrough. The short version:

```bash
py -3.11 -m venv venv
venv\Scripts\python.exe -m pip install -r backend\requirements-full.txt
venv\Scripts\python.exe -m pip install -e ".[dev]"
cd frontend\web && npm install
```

`requirements-lock-windows.txt` is the exact package set the suite was last
verified against, on Windows with CUDA 12.8. On Linux, install CPU torch and
then `pip install -e ".[dev,cv]"`, which is what CI does.

## Before opening a pull request

All of these must pass. CI runs the same checks.

```bash
venv\Scripts\python.exe -m ruff check .
venv\Scripts\python.exe -m pytest -q
cd frontend\web && npm run lint && npm run test:unit && npm run build
```

If you have `samples/sample_15s.mp4` and the trained checkpoints locally, the
suite also runs `tests/test_pipeline_e2e.py`, which fails if pipeline output
changes at all. For an intended change, re-record the fingerprint with
`SSC_UPDATE_PIPELINE_GOLDEN=1` and commit it alongside the change.

mypy runs in CI as a report. Do not add new type errors to files you touch.

## Rules

- **Tests.** Never change the root `conftest.py` or
  `backend/tests/test_database_isolation.py`. They stop the test suite from
  deleting real data. New backend tests follow the same pattern: temp
  `DATABASE_URL`, and only clear the tables the test owns.
- **Imports.** Use absolute imports from the repo root. Do not use
  `sys.path.insert`.
- **Logging.** Library code under `ai/`, `backend/`, and `nexus/` logs through
  `logging` and does not `print`. Ruff enforces this, and the CLI files that are
  allowed to print are listed one by one in `pyproject.toml`.
- **Broad excepts.** `except Exception` needs a `# noqa: BLE001 - <reason>`
  that says why a broad catch is correct there.
- **Paths.** Model and dataset paths go through `configs/registry.py`.
- **Secrets.** Never commit `.env`. Document every new variable in the matching
  `.env.example`.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(tracking): merge re-identified tracks across occlusions
fix(upload): delete partial file when the size limit is hit
docs(run): correct the Celery pool flag for Windows
```

Keep each commit to one logical change. Do not use messages like `update`: a
commit that is 7,000 lines under "update" cannot be reviewed or bisected.
