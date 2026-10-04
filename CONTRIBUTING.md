# Contributing

## Setup

See the [Local Setup](README.md#local-setup) section of the README for environment configuration, Docker Compose services, and running the backend/frontend.

## Before opening a PR

Install the pre-commit hook once, so lint/format issues are caught before they hit CI:

```bash
pip install pre-commit
pre-commit install
```

Run the checks manually at any time with:

```bash
ruff format .
ruff check .
pytest tests/unit/
```

If your change touches chatbot/agent prompts or routing behavior, run the relevant suite in `evaluation/chatbot/` and compare the score against the previous run (see `evaluation/chatbot/README.md`) — these are scored, not pass/fail, and are run manually due to API cost.

If your change touches ML inference/serving code, benchmark it before and after and include the comparison in the PR:

- **Anything on the inference path** (recommendation/similarity routes, `models/` services, pipelines, clients, caching): run the API-level suite in `tests/integration/models/`.
- **Model servers directly** (`model_servers/`, their artifacts or contracts): also run `tests/integration/model_servers/`.

Run the suite on `master` first to get a baseline, then on your branch, and compare the two runs with the suite's `compare_performance.py --auto`. Both suites need a live system (model servers running, artifacts loaded, database populated) and the test user/book IDs described in [Benchmarks](README.md#benchmarks). Standalone scripts (training/data-prep scripts, one-off utilities) don't need this.

## Opening a PR

- Fill out the PR template checklist.
- CI (`backend`, `frontend`) must pass and the PR needs one approving review before it can merge — this repo has branch protection enabled on `master`.
- Keep unrelated changes out of the diff; open a separate PR if you spot something else worth fixing.
