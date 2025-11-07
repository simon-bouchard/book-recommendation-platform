# Docs Agent Evaluation

Two-layer evaluation to verify Docs Agent correctly retrieves and uses documentation.

## Structure

```
evaluation/chatbot/docs_agent/
├── evaluate_docs.py         # Evaluation script
├── docs_test_cases.json     # 7 test cases
└── results/                 # Output (auto-created)
```

## Usage

```bash
python evaluation/chatbot/docs_agent/evaluate_docs.py
```

## Two-Layer Testing

### Layer 1: Document Retrieval (Deterministic)
Validates agent retrieves the correct documents.

**Checks:**
- Did agent call `help_read` tool?
- Were expected document IDs retrieved?

**Pass criteria:** At least one expected document was retrieved

### Layer 2: Answer Quality (LLM-as-judge)
Validates answer correctness and completeness using ground truth.

**Scoring (binary 0/1):**
- **Correctness**: Does response match ground truth information?
- **Completeness**: Does it cover key points from ground truth?

**Pass criteria:** Both scores = 1

## Overall Pass

All three must be true:
1. Agent executed successfully (no errors)
2. Layer 1 passed (correct docs retrieved)
3. Layer 2 passed (quality scores = 1)

## Test Cases

7 common documentation queries:
- How to rate books
- How recommendations work
- How to search
- What is collaborative filtering
- Account creation
- Book subjects
- Recommendation accuracy

Each test case includes:
- `query`: User question
- `expected_docs`: Document aliases that should be retrieved
- `ground_truth`: Expected answer for quality scoring

## Output

Console shows:
- Overall pass rate
- Layer 1 (retrieval) pass rate
- Layer 2 (quality) pass rate
- Per-test results with both layers

JSON saved to `results/` with full details.

## Success Criteria

- Overall pass rate: ≥85% (6/7 tests)
- Layer 1 pass rate: ≥85%
- Layer 2 pass rate: ≥85%

## Notes

**Expected document IDs** are aliases defined in your `docs/help/help_manifest.json`. If you change documentation structure, update `expected_docs` in test cases to match your actual document aliases.

**Ground truth** provides objective criteria for LLM-as-judge. Edit these to match your actual documentation content.
