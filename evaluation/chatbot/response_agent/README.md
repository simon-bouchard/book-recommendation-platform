# Response Agent Evaluation

Simple evaluation to verify Response Agent works correctly.

## Structure

```
evaluation/chatbot/response_agent/
├── evaluate_response.py    # Evaluation script
├── test_cases.json          # 7 test queries
└── results/                 # Output (auto-created)
```

## Usage

```bash
python evaluation/chatbot/response_agent/evaluate_response.py
```

## What It Does

1. Runs Response Agent on 7 basic conversational queries
2. Uses LLM-as-judge to score each response:
   - **Appropriateness** (0 or 1): Does it address the query?
   - **Librarian Tone** (0 or 1): Friendly, helpful, knowledgeable?
3. Pass = both scores = 1

## Test Cases

- Greetings ("Hello")
- Thanks ("Thanks for your help")
- Chitchat ("How are you?")
- General questions ("What can you help me with?")
- Follow-ups with history (2 tests)

## Output

Console shows pass/fail for each test with scores and reasons.
JSON saved to `results/` directory.

## Success Criteria

Pass rate >= 85% (6/7 tests passing)
