# evaluation/chatbot/web_agent/evaluate_web.py
"""
Web Agent evaluation with two-layer testing:
Layer 1 (Deterministic): Web search tool usage validation
Layer 2 (LLM-as-judge): Answer relevance and recency
"""
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.agents.infrastructure.web_agent import WebAgent
from app.agents.domain.entities import AgentRequest, ExecutionContext
from app.agents.logging import capture_agent_console_and_httpx


def load_test_cases(json_path: Path) -> List[Dict]:
    """Load test cases from JSON file."""
    with open(json_path) as f:
        return json.load(f)


def check_web_search_usage(tool_executions: List) -> Dict[str, Any]:
    """
    Layer 1: Check if agent used web_search tool.
    
    Args:
        tool_executions: List of ToolExecution objects from agent
        
    Returns:
        Dict with tool usage validation results
    """
    # Find web_search calls
    web_search_calls = [
        exec for exec in tool_executions 
        if exec.tool_name == "web_search"
    ]
    
    if not web_search_calls:
        return {
            "tool_used": False,
            "reason": "No web_search tool calls made"
        }
    
    return {
        "tool_used": True,
        "reason": f"Made {len(web_search_calls)} web_search call(s)"
    }


def judge_answer_quality(
    query: str,
    response_text: str,
    recency_expected: bool,
    judge_llm
) -> Dict[str, Any]:
    """
    Layer 2: Judge answer relevance and recency.
    
    Scores:
    - Relevance (0/1): Does response address the query?
    - Recency (0/1): Does response contain current/recent information?
    
    Pass = both scores = 1
    """
    # Check for error responses
    if "having trouble" in response_text.lower() or "error:" in response_text.lower():
        return {
            "relevance": 0,
            "recency": 0,
            "reason": "Agent returned error response",
            "quality_pass": False
        }
    
    recency_instruction = ""
    if recency_expected:
        recency_instruction = """
2. RECENCY: Does the response contain current/recent information (2024-2025)?
   - 1 = Yes, references recent information
   - 0 = No, information seems outdated or not time-specific"""
    else:
        recency_instruction = """
2. RECENCY: Not applicable for this query (set to 1 by default)"""
    
    judge_prompt = f"""Query: {query}

Agent Response:
{response_text}

Evaluate the agent's response with binary scoring:

1. RELEVANCE: Does the response properly address the query?
   - 1 = Yes, addresses the query appropriately
   - 0 = No, off-topic or doesn't answer the question

{recency_instruction}

Return JSON:
{{
  "relevance": <0 or 1>,
  "recency": <0 or 1>,
  "reason": "<brief explanation>",
  "pass": <true if both=1, false otherwise>
}}"""

    try:
        with capture_agent_console_and_httpx():
            response = judge_llm.invoke([{"role": "user", "content": judge_prompt}])
        content = response.content if hasattr(response, 'content') else str(response)
        
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        result = json.loads(content.strip())
        
        # If recency not expected, auto-set to 1
        if not recency_expected:
            result["recency"] = 1
        
        return {
            "relevance": result.get("relevance", 0),
            "recency": result.get("recency", 0),
            "reason": result.get("reason", ""),
            "quality_pass": result.get("pass", False)
        }
    except Exception as e:
        return {
            "relevance": 0,
            "recency": 0,
            "reason": f"Judge failed: {str(e)}",
            "quality_pass": False
        }


def evaluate(test_cases: List[Dict], agent: WebAgent) -> Dict[str, Any]:
    """Evaluate web agent on test cases."""
    results = []
    judge_llm = agent.llm  # Reuse agent's LLM for judging
    
    for test in test_cases:
        query = test["query"]
        recency_expected = test.get("recency_expected", False)
        
        # Execute agent
        request = AgentRequest(
            user_text=query,
            conversation_history=[],
            context=ExecutionContext()
        )
        
        try:
            response = agent.execute(request)
            response_text = response.text
            agent_success = response.success
            tool_executions = response.execution_state.tool_executions if response.execution_state else []
            
            # Check for error patterns
            if not agent_success or "having trouble" in response_text.lower():
                agent_success = False
                
        except Exception as e:
            response_text = f"ERROR: {str(e)}"
            agent_success = False
            tool_executions = []
        
        # Layer 1: Tool usage validation
        tool_check = check_web_search_usage(tool_executions)
        
        # Layer 2: Answer quality
        quality_check = judge_answer_quality(query, response_text, recency_expected, judge_llm)
        
        # Overall pass: agent success AND tool used AND quality good
        overall_pass = (
            agent_success and 
            tool_check["tool_used"] and 
            quality_check["quality_pass"]
        )
        
        results.append({
            "query": query,
            "response": response_text,
            "agent_success": agent_success,
            "recency_expected": recency_expected,
            # Layer 1
            "tool_used": tool_check["tool_used"],
            "tool_reason": tool_check["reason"],
            # Layer 2
            "relevance": quality_check["relevance"],
            "recency": quality_check["recency"],
            "quality_reason": quality_check["reason"],
            "quality_pass": quality_check["quality_pass"],
            # Overall
            "overall_pass": overall_pass
        })
    
    # Stats
    passed = sum(1 for r in results if r["overall_pass"])
    tool_passed = sum(1 for r in results if r["tool_used"])
    quality_passed = sum(1 for r in results if r["quality_pass"])
    total = len(results)
    
    return {
        "results": results,
        "passed": passed,
        "total": total,
        "pass_rate": passed / total if total > 0 else 0,
        "tool_pass_rate": tool_passed / total if total > 0 else 0,
        "quality_pass_rate": quality_passed / total if total > 0 else 0,
        "timestamp": datetime.now().isoformat()
    }


def print_results(eval_results: Dict[str, Any]):
    """Print results to console."""
    print("\n" + "="*70)
    print("WEB AGENT EVALUATION")
    print("="*70)
    
    print(f"\nOverall Pass Rate: {eval_results['pass_rate']:.1%} ({eval_results['passed']}/{eval_results['total']})")
    print(f"Layer 1 (Tool Usage): {eval_results['tool_pass_rate']:.1%}")
    print(f"Layer 2 (Quality): {eval_results['quality_pass_rate']:.1%}")
    
    # Show each test
    print("\nTest Results:")
    for i, r in enumerate(eval_results["results"], 1):
        status = "✅ PASS" if r["overall_pass"] else "❌ FAIL"
        print(f"\n{i}. {status}")
        print(f"   Query: {r['query']}")
        print(f"   Response: {r['response'][:100]}...")
        
        # Layer 1 results
        tool_status = "✅" if r["tool_used"] else "❌"
        print(f"   {tool_status} Layer 1 (Tool Usage): {r['tool_reason']}")
        
        # Layer 2 results
        qual_status = "✅" if r["quality_pass"] else "❌"
        recency_note = " (recency checked)" if r["recency_expected"] else ""
        print(f"   {qual_status} Layer 2 (Quality): Relevance={r['relevance']}, Recency={r['recency']}{recency_note}")
        
        # Show reasons for failures
        if not r["tool_used"]:
            print(f"      Tool: {r['tool_reason']}")
        if not r["quality_pass"]:
            print(f"      Quality: {r['quality_reason']}")
    
    print("\n" + "="*70)


def save_results(eval_results: Dict[str, Any], output_dir: Path):
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"web_eval_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def main():
    script_dir = Path(__file__).parent
    test_cases_path = script_dir / "web_test_cases.json"
    results_dir = script_dir / "results"
    
    print("Loading test cases...")
    test_cases = load_test_cases(test_cases_path)
    print(f"Loaded {len(test_cases)} test cases")
    
    print("Initializing Web Agent...")
    try:
        agent = WebAgent()
        print("✓ Agent initialized successfully\n")
    except Exception as e:
        print(f"\n❌ ERROR: Failed to initialize Web Agent")
        print(f"   {str(e)}")
        return
    
    print("Running evaluation...\n")
    eval_results = evaluate(test_cases, agent)
    
    print_results(eval_results)
    save_results(eval_results, results_dir)


if __name__ == "__main__":
    main()
