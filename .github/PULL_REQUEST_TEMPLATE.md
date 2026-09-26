## Summary

<!-- What does this change do, and why? -->

## Testing

<!-- How did you verify this? Commands run, manual steps, screenshots, etc. -->

## Checklist

- [ ] `ruff format .` and `ruff check .` pass locally (or `pre-commit run --all-files`)
- [ ] `pytest tests/unit/` passes
- [ ] If this touches chatbot/agent prompts or routing behavior: agent evaluations (`evaluation/chatbot/`) run and score hasn't regressed vs. the previous run
- [ ] If this touches ML inference/serving code (model servers, live pipelines): integration tests (`tests/integration/model_servers/`, `tests/integration/models/`) pass
- [ ] No unrelated files included in the diff

Integration tests aren't required for standalone scripts (training/data-prep scripts, one-off utilities) that don't touch the live inference path.
