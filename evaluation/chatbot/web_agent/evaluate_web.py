# evaluation/chatbot/web_agent/evaluate_web.py
"""
Web Agent evaluation with explicit tool requirement control.
Determines agent success based on actual output quality rather than internal success flag.
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


def check_agent_success(response_text: str) -> tuple[bool, str]:
    """
    Determine if agent execution was successful based on response text.
    
    Don't rely on response.success flag - agent may mark itself as failed
    even when producing good output. Instead, check the actual response.
    
    Returns:
        (success, reason)
    """
    # Check for explicit error patterns
    error_indicators = [
        "error:",
        "exception:",
        "failed to",
        "couldn't find",
        "unable to",
        "don't have the information",
        "training data only",
        "has not yet occurred"
    ]
    
    lower_text = response_text.lower()
    
    # Check for error indicators
    for indicator in error_indicators:
        if indicator in lower_text:
            return False, f"Response contains error indicator: '{indicator}'"
    
    # Check for minimum substance
    if len(response_text.strip()) < 30:
        return False, "Response too short (< 30 chars)"
    
    # Check for cop-out responses
    cop_out_phrases = [
        "having trouble",
        "couldn't summarize",
        "check these sources",
        "see the following links"
    ]
    
    for phrase in cop_out_phrases:
        if phrase in lower_text and len(response_text) < 200:
            return False, f"Response is a cop-out: contains '{phrase}'"
    
    return True, "Response is substantive"


def check_web_search_usage(tool_executions: List) -> Dict[str, Any]:
    """
    Check if agent used web_search tool.
    
    Args:
        tool_executions: List of ToolExecution objects from agent
        
    Returns:
        Dict with tool usage validation results
    """
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
    Judge answer relevance and recency using LLM.
    
    Scores:
    - Relevance (0/1): Does response address the query?
    - Recency (0/1): Does response contain current/recent information?
    
    Pass = both scores = 1
    """
    recency_instruction = ""
    if recency_expected:
        recency_instruction = """
2. RECENCY: Does the response contain current/recent information (2024-2025)?
   - 1 = Yes, references recent information with dates/specifics
   - 0 = No, information seems outdated or lacks temporal specificity"""
    else:
        recency_instruction = """
2. RECENCY: Not applicable for this query (automatically set to 1)"""
    
    judge_prompt = f"""Query: {query}

Agent Response:
{response_text}

Evaluate the agent's response with binary scoring:

1. RELEVANCE: Does the response properly address the query?
   - 1 = Yes, directly addresses the query
   - 0 = No, off-topic or doesn't answer

{recency_instruction}

Return ONLY valid JSON:
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
        
        # Strip markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        result = json.loads(content.strip())
        
        # Auto-set recency to 1 if not expected
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
    judge_llm = agent.llm
    
    for test in test_cases:
        query = test["query"]
        recency_expected = test.get("recency_expected", False)
        tool_required = test.get("tool_required", True)
        rationale = test.get("rationale", "")
        
        # Execute agent
        request = AgentRequest(
            user_text=query,
            conversation_history=[],
            context=ExecutionContext()
        )
        
        try:
            response = agent.execute(request)
            response_text = response.text
            tool_executions = response.execution_state.tool_executions if response.execution_state else []
        except Exception as e:
            response_text = f"EXCEPTION: {str(e)}"
            tool_executions = []
        
        # Determine agent success based on response quality, not internal flag
        agent_success, success_reason = check_agent_success(response_text)
        
        # Layer 1: Tool usage validation
        tool_check = check_web_search_usage(tool_executions)
        
        # Layer 2: Answer quality
        quality_check = judge_answer_quality(query, response_text, recency_expected, judge_llm)
        
        # Overall pass logic based on tool_required flag
        if tool_required:
            overall_pass = (
                agent_success and 
                tool_check["tool_used"] and 
                quality_check["quality_pass"]
            )
            pass_logic = "Tool required: web search + quality answer"
            
            if not overall_pass:
                if not agent_success:
                    failure_reason = f"Agent failed: {success_reason}"
                elif not tool_check["tool_used"]:
                    failure_reason = "Required web search not used"
                elif not quality_check["quality_pass"]:
                    failure_reason = f"Quality failed: {quality_check['reason']}"
                else:
                    failure_reason = "Unknown"
            else:
                failure_reason = None
        else:
            overall_pass = (
                agent_success and 
                quality_check["quality_pass"]
            )
            pass_logic = "Tool optional: quality answer sufficient"
            
            if not overall_pass:
                if not agent_success:
                    failure_reason = f"Agent failed: {success_reason}"
                elif not quality_check["quality_pass"]:
                    failure_reason = f"Quality failed: {quality_check['reason']}"
                else:
                    failure_reason = "Unknown"
            else:
                failure_reason = None
        
        results.append({
            "query": query,
            "response": response_text,
            "rationale": rationale,
            "agent_success": agent_success,
            "agent_success_reason": success_reason,
            "recency_expected": recency_expected,
            "tool_required": tool_required,
            "pass_logic": pass_logic,
            # Layer 1
            "tool_used": tool_check["tool_used"],
            "tool_reason": tool_check["reason"],
            # Layer 2
            "relevance": quality_check["relevance"],
            "recency": quality_check["recency"],
            "quality_reason": quality_check["reason"],
            "quality_pass": quality_check["quality_pass"],
            # Overall
            "overall_pass": overall_pass,
            "failure_reason": failure_reason
        })
    
    # Stats
    passed = sum(1 for r in results if r["overall_pass"])
    tool_passed = sum(1 for r in results if r["tool_used"])
    quality_passed = sum(1 for r in results if r["quality_pass"])
    agent_success_count = sum(1 for r in results if r["agent_success"])
    
    # Breakdown by tool requirement
    tool_required_tests = [r for r in results if r["tool_required"]]
    tool_optional_tests = [r for r in results if not r["tool_required"]]
    
    tool_required_passed = sum(1 for r in tool_required_tests if r["overall_pass"])
    tool_optional_passed = sum(1 for r in tool_optional_tests if r["overall_pass"])
    
    total = len(results)
    
    return {
        "results": results,
        "passed": passed,
        "total": total,
        "pass_rate": passed / total if total > 0 else 0,
        "agent_success_rate": agent_success_count / total if total > 0 else 0,
        "tool_pass_rate": tool_passed / total if total > 0 else 0,
        "quality_pass_rate": quality_passed / total if total > 0 else 0,
        "tool_required_pass_rate": tool_required_passed / len(tool_required_tests) if tool_required_tests else 0,
        "tool_optional_pass_rate": tool_optional_passed / len(tool_optional_tests) if tool_optional_tests else 0,
        "breakdown": {
            "tool_required_tests": len(tool_required_tests),
            "tool_required_passed": tool_required_passed,
            "tool_optional_tests": len(tool_optional_tests),
            "tool_optional_passed": tool_optional_passed
        },
        "timestamp": datetime.now().isoformat()
    }


def print_results(eval_results: Dict[str, Any]):
    """Print results to console."""
    print("\n" + "="*70)
    print("WEB AGENT EVALUATION")
    print("="*70)
    
    print(f"\nOverall Pass Rate: {eval_results['pass_rate']:.1%} ({eval_results['passed']}/{eval_results['total']})")
    print(f"Agent Success Rate: {eval_results['agent_success_rate']:.1%}")
    print(f"Layer 1 (Tool Usage): {eval_results['tool_pass_rate']:.1%}")
    print(f"Layer 2 (Quality): {eval_results['quality_pass_rate']:.1%}")
    
    breakdown = eval_results['breakdown']
    print(f"\nBreakdown by Tool Requirement:")
    print(f"  Tool-Required: {eval_results['tool_required_pass_rate']:.1%} ({breakdown['tool_required_passed']}/{breakdown['tool_required_tests']})")
    print(f"  Tool-Optional: {eval_results['tool_optional_pass_rate']:.1%} ({breakdown['tool_optional_passed']}/{breakdown['tool_optional_tests']})")
    
    print("\nTest Results:")
    for i, r in enumerate(eval_results["results"], 1):
        status = "✅ PASS" if r["overall_pass"] else "❌ FAIL"
        print(f"\n{i}. {status}")
        print(f"   Query: {r['query']}")
        print(f"   Rationale: {r['rationale']}")
        print(f"   Response: {r['response'][:100]}...")
        
        # Agent success
        agent_icon = "✅" if r["agent_success"] else "❌"
        print(f"   {agent_icon} Agent: {r['agent_success_reason']}")
        
        # Tool usage
        requirement = "REQUIRED" if r["tool_required"] else "OPTIONAL"
        tool_icon = "✅" if r["tool_used"] else ("❌" if r["tool_required"] else "ℹ️")
        print(f"   {tool_icon} Layer 1 (Tool - {requirement}): {r['tool_reason']}")
        
        # Quality
        qual_icon = "✅" if r["quality_pass"] else "❌"
        recency_note = " (recency checked)" if r["recency_expected"] else ""
        print(f"   {qual_icon} Layer 2 (Quality): Rel={r['relevance']}, Rec={r['recency']}{recency_note}")
        
        # Failure reason
        if not r["overall_pass"]:
            print(f"   ⚠️  {r['failure_reason']}")
    
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
    
    tool_required = sum(1 for t in test_cases if t.get("tool_required", True))
    tool_optional = len(test_cases) - tool_required
    print(f"  - {tool_required} tool-required tests")
    print(f"  - {tool_optional} tool-optional tests")
    
    print("\nInitializing Web Agent...")
    try:
        agent = WebAgent()
        print("✓ Agent initialized\n")
    except Exception as e:
        print(f"\n❌ Failed to initialize: {e}")
        return
    
    print("Running evaluation...\n")
    eval_results = evaluate(test_cases, agent)
    
    print_results(eval_results)
    save_results(eval_results, results_dir)


if __name__ == "__main__":
    main()
