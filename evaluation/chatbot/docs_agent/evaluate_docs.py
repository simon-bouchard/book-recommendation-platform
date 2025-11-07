# evaluation/chatbot/docs_agent/evaluate_docs.py
"""
Docs Agent evaluation with two-layer testing:
Layer 1 (Deterministic): Document retrieval validation
Layer 2 (LLM-as-judge): Answer quality with ground truth
"""
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.agents.infrastructure.docs_agent import DocsAgent
from app.agents.domain.entities import AgentRequest, ExecutionContext
from app.agents.logging import capture_agent_console_and_httpx


def load_test_cases(json_path: Path) -> List[Dict]:
    """Load test cases from JSON file."""
    with open(json_path) as f:
        return json.load(f)


def check_document_retrieval(
    tool_executions: List,
    expected_docs: List[str]
) -> Dict[str, Any]:
    """
    Layer 1: Check if agent retrieved expected documents.
    
    Args:
        tool_executions: List of ToolExecution objects from agent
        expected_docs: List of expected document aliases/filenames
        
    Returns:
        Dict with retrieval validation results
    """
    # Find all help_read calls
    help_read_calls = [
        exec for exec in tool_executions 
        if exec.tool_name == "help_read"
    ]
    
    if not help_read_calls:
        return {
            "retrieved_docs": [],
            "expected_docs": expected_docs,
            "retrieval_pass": False,
            "reason": "No help_read tool calls made"
        }
    
    # Extract doc names from arguments
    retrieved_docs = []
    for call in help_read_calls:
        doc_name = call.arguments.get("doc_name", "")
        if doc_name:
            retrieved_docs.append(doc_name.lower())
    
    # Check if expected docs were retrieved
    expected_set = {doc.lower() for doc in expected_docs}
    retrieved_set = set(retrieved_docs)
    
    # Pass if any expected doc was retrieved
    retrieval_pass = bool(expected_set & retrieved_set)
    
    if retrieval_pass:
        reason = f"Retrieved expected docs: {expected_set & retrieved_set}"
    else:
        reason = f"Expected {expected_docs} but got {retrieved_docs}"
    
    return {
        "retrieved_docs": retrieved_docs,
        "expected_docs": expected_docs,
        "retrieval_pass": retrieval_pass,
        "reason": reason
    }


def judge_answer_quality(
    query: str,
    response_text: str,
    ground_truth: str,
    judge_llm
) -> Dict[str, Any]:
    """
    Layer 2: Judge answer quality against ground truth.
    
    Scores:
    - Correctness (0/1): Does response match expected information?
    - Completeness (0/1): Does it cover key points from ground truth?
    
    Pass = both scores = 1
    """
    # Check for error responses
    if "having trouble" in response_text.lower() or "error:" in response_text.lower():
        return {
            "correctness": 0,
            "completeness": 0,
            "reason": "Agent returned error response",
            "quality_pass": False
        }
    
    judge_prompt = f"""Query: {query}

Expected Answer (Ground Truth):
{ground_truth}

Agent Response:
{response_text}

Evaluate the agent's response with binary scoring:

1. CORRECTNESS: Does the response provide accurate information matching the ground truth?
   - 1 = Yes, information is correct and matches ground truth
   - 0 = No, information is wrong or contradicts ground truth

2. COMPLETENESS: Does the response cover the key points from the ground truth?
   - 1 = Yes, covers main points adequately
   - 0 = No, missing important information

Return JSON:
{{
  "correctness": <0 or 1>,
  "completeness": <0 or 1>,
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
        
        return {
            "correctness": result.get("correctness", 0),
            "completeness": result.get("completeness", 0),
            "reason": result.get("reason", ""),
            "quality_pass": result.get("pass", False)
        }
    except Exception as e:
        return {
            "correctness": 0,
            "completeness": 0,
            "reason": f"Judge failed: {str(e)}",
            "quality_pass": False
        }


def evaluate(test_cases: List[Dict], agent: DocsAgent) -> Dict[str, Any]:
    """Evaluate docs agent on test cases."""
    results = []
    judge_llm = agent.llm  # Reuse agent's LLM for judging
    
    for test in test_cases:
        query = test["query"]
        expected_docs = test.get("expected_docs", [])
        ground_truth = test.get("ground_truth", "")
        
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
        
        # Layer 1: Document retrieval validation
        retrieval_check = check_document_retrieval(tool_executions, expected_docs)
        
        # Layer 2: Answer quality with ground truth
        quality_check = judge_answer_quality(query, response_text, ground_truth, judge_llm)
        
        # Overall pass: agent success AND retrieval correct AND quality good
        overall_pass = (
            agent_success and 
            retrieval_check["retrieval_pass"] and 
            quality_check["quality_pass"]
        )
        
        results.append({
            "query": query,
            "response": response_text,
            "agent_success": agent_success,
            # Layer 1
            "retrieved_docs": retrieval_check["retrieved_docs"],
            "expected_docs": retrieval_check["expected_docs"],
            "retrieval_pass": retrieval_check["retrieval_pass"],
            "retrieval_reason": retrieval_check["reason"],
            # Layer 2
            "correctness": quality_check["correctness"],
            "completeness": quality_check["completeness"],
            "quality_reason": quality_check["reason"],
            "quality_pass": quality_check["quality_pass"],
            # Overall
            "overall_pass": overall_pass
        })
    
    # Stats
    passed = sum(1 for r in results if r["overall_pass"])
    retrieval_passed = sum(1 for r in results if r["retrieval_pass"])
    quality_passed = sum(1 for r in results if r["quality_pass"])
    total = len(results)
    
    return {
        "results": results,
        "passed": passed,
        "total": total,
        "pass_rate": passed / total if total > 0 else 0,
        "retrieval_pass_rate": retrieval_passed / total if total > 0 else 0,
        "quality_pass_rate": quality_passed / total if total > 0 else 0,
        "timestamp": datetime.now().isoformat()
    }


def print_results(eval_results: Dict[str, Any]):
    """Print results to console."""
    print("\n" + "="*70)
    print("DOCS AGENT EVALUATION")
    print("="*70)
    
    print(f"\nOverall Pass Rate: {eval_results['pass_rate']:.1%} ({eval_results['passed']}/{eval_results['total']})")
    print(f"Layer 1 (Retrieval): {eval_results['retrieval_pass_rate']:.1%}")
    print(f"Layer 2 (Quality): {eval_results['quality_pass_rate']:.1%}")
    
    # Show each test
    print("\nTest Results:")
    for i, r in enumerate(eval_results["results"], 1):
        status = "✅ PASS" if r["overall_pass"] else "❌ FAIL"
        print(f"\n{i}. {status}")
        print(f"   Query: {r['query']}")
        print(f"   Response: {r['response'][:100]}...")
        
        # Layer 1 results
        ret_status = "✅" if r["retrieval_pass"] else "❌"
        print(f"   {ret_status} Layer 1 (Retrieval): Expected {r['expected_docs']}, Got {r['retrieved_docs']}")
        
        # Layer 2 results
        qual_status = "✅" if r["quality_pass"] else "❌"
        print(f"   {qual_status} Layer 2 (Quality): Correctness={r['correctness']}, Completeness={r['completeness']}")
        
        # Show reasons for failures
        if not r["retrieval_pass"]:
            print(f"      Retrieval: {r['retrieval_reason']}")
        if not r["quality_pass"]:
            print(f"      Quality: {r['quality_reason']}")
    
    print("\n" + "="*70)


def save_results(eval_results: Dict[str, Any], output_dir: Path):
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"docs_eval_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def main():
    script_dir = Path(__file__).parent
    test_cases_path = script_dir / "docs_test_cases.json"
    results_dir = script_dir / "results"
    
    print("Loading test cases...")
    test_cases = load_test_cases(test_cases_path)
    print(f"Loaded {len(test_cases)} test cases")
    
    print("Initializing Docs Agent...")
    try:
        agent = DocsAgent()
        print("✓ Agent initialized successfully\n")
    except Exception as e:
        print(f"\n❌ ERROR: Failed to initialize Docs Agent")
        print(f"   {str(e)}")
        return
    
    print("Running evaluation...\n")
    eval_results = evaluate(test_cases, agent)
    
    print_results(eval_results)
    save_results(eval_results, results_dir)


if __name__ == "__main__":
    main()
