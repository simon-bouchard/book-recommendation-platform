# Evaluation Dashboard

A comprehensive dashboard for monitoring chatbot agent evaluation results across all agents.

## Features

- **Summary View**: Quick overview of all agents with pass rates and status
- **Detailed Reports**: Comprehensive breakdown of each agent's performance
- **Failed Test Analysis**: Detailed information about what went wrong
- **Historical Comparison**: Compare current results with previous runs
- **Flexible Detection**: Works with both standard directory structure and flat file layouts

## Usage

### Basic Usage (Summary Only)

```bash
python eval_dashboard.py
```

Shows a summary table of all agents with pass rates and recommendations.

### Verbose Mode (Detailed Breakdown)

```bash
python eval_dashboard.py -v
```

Shows detailed breakdown for each agent including:
- Overall performance metrics
- Category-by-category breakdown
- Failed test details with reasons
- Retrieval/quality metrics where applicable

### Custom Path

```bash
python eval_dashboard.py -p /path/to/evaluation
```

Specify a custom base directory for evaluation results.

### Disable Comparison

```bash
python eval_dashboard.py --no-comparison
```

Don't compare with previous evaluation runs (only show latest).

## Example Output

```
================================================================================
  🎯 EVALUATION DASHBOARD SUMMARY
================================================================================

Agent                Pass Rate       Tests        Status
--------------------------------------------------------------------------------
web                  50.0%           4/8          ⚠️  Needs Work
recommendation       70.6%           12/17        ⚠️  Needs Work
docs                 75.0%           6/8          ⚠️  Needs Work
router               100.0%          20/20        ✅ Perfect
router_history       100.0%          16/16        ✅ Perfect
response             100.0%          7/7          ✅ Perfect
--------------------------------------------------------------------------------

💡 Priority Recommendations:
   🟡 Needs Improvement: web, recommendation, docs
```

## Directory Structure

The script expects evaluation results in the following structure:

```
evaluation/chatbot/
├── router/results/
│   ├── router_eval_*.json
│   └── router_history_eval_*.json
├── docs_agent/results/
│   └── docs_eval_*.json
├── web_agent/results/
│   └── web_eval_*.json
├── response_agent/results/
│   └── response_eval_*.json
└── recommendation_agent/results/
    └── recommendation_eval_*.json
```

The script automatically finds the latest eval file for each agent based on timestamp.

## Shared Utilities: `eval_utils.py`

### Overview

The `evaluation/chatbot/eval_utils.py` module provides shared utilities for all agent evaluation scripts. This eliminates code duplication and ensures consistent behavior across all evaluations.

### `execute_with_streaming()`

The primary utility is the `execute_with_streaming()` helper function, which handles async streaming execution for agents.

**Purpose:**
- Executes agents using their `execute_stream()` method
- Accumulates streamed text chunks
- Reconstructs full `AgentResponse` from streaming chunks
- Converts tool calls to `ToolExecution` objects
- Makes streaming transparent to evaluation logic

**Usage:**

```python
from evaluation.chatbot.eval_utils import execute_with_streaming

async def evaluate(test_cases, agent):
    for test in test_cases:
        request = AgentRequest(user_text=test["query"], ...)

        # Execute with streaming - returns complete AgentResponse
        response = await execute_with_streaming(agent, request)

        # Now work with full response as if using sync execute()
        print(response.text)
        print(response.execution_state.tool_executions)
```

**Benefits:**
- **DRY**: Single implementation shared across all evaluation scripts
- **Consistency**: Identical streaming behavior for all agents
- **Maintainability**: Bug fixes apply to all evaluations automatically
- **Simplicity**: Evaluation scripts focus on validation logic, not streaming mechanics

**Alternative: Sync Execution**

If you don't need to test streaming behavior, agents also support direct async execution:

```python
# Direct async execution (no streaming)
response = await agent.execute(request)
```

Both approaches return the same `AgentResponse` structure with identical fields.

## Supported Agents

- **Router**: Query routing decisions (with and without history)
- **Docs**: Documentation retrieval and response quality
- **Web**: Web search tool usage and response quality
- **Response**: Conversational response appropriateness
- **Recommendation**: Book recommendation retrieval and quality

## Status Indicators

- ✅ **Perfect**: 100% pass rate
- ⚠️  **Good**: 80-99% pass rate
- ⚠️  **Needs Work**: 50-79% pass rate
- ❌ **Critical**: <50% pass rate
