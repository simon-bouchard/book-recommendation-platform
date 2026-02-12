# evaluation/chatbot/response_agent/evaluate_response.py
"""
Response Agent evaluation - Simple test to verify it works.
Tests basic conversational queries with LLM-as-judge.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.agents.infrastructure.response_agent import ResponseAgent
from app.agents.domain.entities import AgentRequest, ExecutionContext
from app.agents.settings import get_llm
from app.agents.logging import capture_agent_console_and_httpx

# Import shared streaming helper
from evaluation.chatbot.eval_utils import execute_with_streaming


def load_test_cases(json_path: Path) -> List[Dict]:
    """Load test cases from JSON file."""
    with open(json_path) as f:
        return json.load(f)


def judge_response(
    query: str, response_text: str, history: List[Dict], judge_llm
) -> Dict[str, Any]:
    """
    LLM-as-judge scores response on:
    - Appropriateness (0 or 1): Does it properly address the query?
    - Librarian tone (0 or 1): Friendly, helpful, knowledgeable?

    Pass = both scores = 1
    """
    # Check for error responses first
    if "having trouble" in response_text.lower() or "error" in response_text.lower():
        return {
            "appropriateness": 0,
            "tone": 0,
            "reason": "Agent returned error response",
            "pass": False,
        }

    history_text = ""
    if history:
        history_text = "Conversation History:\n"
        for turn in history[-3:]:
            history_text += f"User: {turn.get('u', '')}\nAssistant: {turn.get('a', '')}\n"

    judge_prompt = f"""{history_text}
User Query: {query}
Agent Response: {response_text}

Evaluate this response with binary scoring:

1. APPROPRIATENESS: Does it properly address the user's query?
   - 1 = Yes, addresses the query appropriately
   - 0 = No, off-topic or doesn't address the query

2. LIBRARIAN TONE: Is it friendly, helpful, and knowledgeable?
   - 1 = Yes, has good librarian tone
   - 0 = No, cold/robotic/inappropriate tone

Return JSON:
{{
  "appropriateness": <0 or 1>,
  "tone": <0 or 1>,
  "reason": "<brief explanation>",
  "pass": <true if both=1, false otherwise>
}}"""

    try:
        with capture_agent_console_and_httpx():
            response = judge_llm.invoke([{"role": "user", "content": judge_prompt}])
        content = response.content if hasattr(response, "content") else str(response)

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        result = json.loads(content.strip())

        return {
            "appropriateness": result.get("appropriateness", 0),
            "tone": result.get("tone", 0),
            "reason": result.get("reason", ""),
            "pass": result.get("pass", False),
        }
    except Exception as e:
        return {"appropriateness": 0, "tone": 0, "reason": f"Judge failed: {str(e)}", "pass": False}


async def evaluate(test_cases: List[Dict], agent: ResponseAgent, judge_llm) -> Dict[str, Any]:
    """Evaluate response agent on test cases."""
    results = []

    for test in test_cases:
        query = test["query"]
        history = test.get("history", [])

        # Execute agent using shared streaming helper
        request = AgentRequest(
            user_text=query, conversation_history=history, context=ExecutionContext()
        )

        try:
            # Use shared streaming helper
            response = await execute_with_streaming(agent, request)
            response_text = response.text
            agent_success = response.success

            # Check for known error patterns
            if (
                not agent_success
                or "having trouble" in response_text.lower()
                or "error:" in response_text.lower()
            ):
                agent_success = False

        except Exception as e:
            response_text = f"ERROR: {str(e)}"
            agent_success = False

        # Judge the response (will auto-fail if error detected)
        scores = judge_response(query, response_text, history, judge_llm)

        results.append(
            {
                "query": query,
                "history": history,
                "response": response_text,
                "agent_success": agent_success,
                "appropriateness": scores["appropriateness"],
                "tone": scores["tone"],
                "reason": scores["reason"],
                "pass": scores["pass"] and agent_success,  # Both must be true
            }
        )

    # Stats
    passed = sum(1 for r in results if r["pass"])
    total = len(results)

    return {
        "results": results,
        "passed": passed,
        "total": total,
        "pass_rate": passed / total if total > 0 else 0,
        "timestamp": datetime.now().isoformat(),
    }


def print_results(eval_results: Dict[str, Any]):
    """Print results to console."""
    print("\n" + "=" * 70)
    print("RESPONSE AGENT EVALUATION")
    print("=" * 70)

    print(
        f"\nPass Rate: {eval_results['pass_rate']:.1%} ({eval_results['passed']}/{eval_results['total']})"
    )

    # Show each test
    print("\nTest Results:")
    for i, r in enumerate(eval_results["results"], 1):
        status = "✅ PASS" if r["pass"] else "❌ FAIL"
        print(f"\n{i}. {status}")
        print(f"   Query: {r['query']}")
        if r["history"]:
            print(f"   (has history: {len(r['history'])} turns)")
        print(f"   Response: {r['response'][:100]}...")
        print(f"   Scores: Appropriateness={r['appropriateness']}, Tone={r['tone']}")
        if not r["pass"] or not r["agent_success"]:
            print(f"   Reason: {r['reason']}")

    print("\n" + "=" * 70)


def save_results(eval_results: Dict[str, Any], output_dir: Path):
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"response_eval_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


async def async_main():
    """Async entry point for evaluation."""
    script_dir = Path(__file__).parent
    test_cases_path = script_dir / "test_cases.json"
    results_dir = script_dir / "results"

    print("Loading test cases...")
    test_cases = load_test_cases(test_cases_path)
    print(f"Loaded {len(test_cases)} test cases")

    print("Initializing agent and judge...")
    try:
        agent = ResponseAgent()
        judge_llm = get_llm(tier="small", json_mode=True, temperature=0, timeout=15)

        # Quick health check
        print("Testing LLM connection...")
        with capture_agent_console_and_httpx():
            test_response = judge_llm.invoke([{"role": "user", "content": "Say 'OK'"}])
        print("✓ LLM connection successful\n")
    except Exception as e:
        print(f"\n❌ ERROR: Failed to initialize LLM")
        print(f"   {str(e)}")
        print("\nPossible issues:")
        print("- Missing or invalid API key")
        print("- Network connectivity")
        print("- Check your environment variables and settings\n")
        return

    print("Running evaluation...\n")
    eval_results = await evaluate(test_cases, agent, judge_llm)

    print_results(eval_results)
    save_results(eval_results, results_dir)


def main():
    """Synchronous entry point that runs async code."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
