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

If your change touches ML inference/serving code (model servers, live pipelines), run the relevant suite in `tests/integration/model_servers/` or `tests/integration/models/`. These require a live system (model servers running, artifacts loaded) and are not required for standalone scripts (training/data-prep scripts, one-off utilities).

## Opening a PR

- Fill out the PR template checklist.
- CI (`backend`, `frontend`) must pass and the PR needs one approving review before it can merge — this repo has branch protection enabled on `master`.
- Keep unrelated changes out of the diff; open a separate PR if you spot something else worth fixing.
