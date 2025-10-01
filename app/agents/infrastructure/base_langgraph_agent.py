# app/agents/infrastructure/base_langgraph_agent.py
"""
Base LangGraph agent implementing ReAct pattern with message-based prompting.
"""
import json
import time
from typing import Dict, Any, List, Optional
from abc import abstractmethod
import threading

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.agents.domain.entities import (
    AgentExecutionState,
    AgentRequest,
    AgentResponse,
    ExecutionStatus,
)
from app.agents.domain.interfaces import BaseAgent
from app.agents.domain.services import StandardResultProcessor
from app.agents.settings import get_llm
from app.agents.tools.registry import ToolRegistry, InternalToolGates
from app.agents.logging import append_chatbot_log
from .tool_executor import ToolExecutor


class TimeoutException(Exception):
    """Raised when execution exceeds timeout."""
    pass


def run_with_timeout(func, timeout_seconds: int, *args, **kwargs):
    """Run a function with a timeout using threading."""
    result_container = {'result': None, 'exception': None, 'completed': False}
    
    def target():
        try:
            result_container['result'] = func(*args, **kwargs)
            result_container['completed'] = True
        except Exception as e:
            result_container['exception'] = e
            result_container['completed'] = True
    
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    if not result_container['completed']:
        raise TimeoutException(f"Execution exceeded {timeout_seconds}s timeout")
    
    if result_container['exception']:
        raise result_container['exception']
    
    return result_container['result']


class BaseLangGraphAgent(BaseAgent):
    """
    Base agent implementing ReAct loop with message-based prompting.
    
    Graph flow:
        reason → [act | finalize]
          ↑        ↓
          └────────┘
    """
    
    def __init__(self, configuration, ctx_user=None, ctx_db=None):
        super().__init__(configuration)
        
        self.llm = get_llm(tier=configuration.llm_tier, temperature=0.0)

        # Tool system
        self.registry = self._create_tool_registry(ctx_user, ctx_db)
        self.tool_executor = ToolExecutor(self.registry)
        
        # Result processor
        self.result_processor = StandardResultProcessor()
        
        # Build the graph
        self.graph = self._build_graph()
        
        append_chatbot_log(
            f"Initialized {self.__class__.__name__} with "
            f"{len(self.tool_executor._tools)} tools"
        )
    
    def _create_tool_registry(self, ctx_user, ctx_db) -> ToolRegistry:
        """Create tool registry based on agent capabilities."""
        config = self.configuration
        
        gates = InternalToolGates(
            user_num_ratings=0,
            warm_threshold=10,
            profile_allowed=False
        )
        
        from app.agents.domain.entities import AgentCapability
        
        return ToolRegistry(
            web=AgentCapability.WEB_SEARCH in config.capabilities,
            docs=AgentCapability.DOCUMENT_SEARCH in config.capabilities,
            internal=AgentCapability.INTERNAL_TOOLS in config.capabilities,
            gates=gates,
            ctx_user=ctx_user,
            ctx_db=ctx_db,
        )
    
    def _build_graph(self) -> StateGraph:
        """Build simplified ReAct loop graph."""
        workflow = StateGraph(AgentExecutionState)
        
        workflow.add_node("reason", self._reason_node)
        workflow.add_node("act", self._act_node)
        workflow.add_node("finalize", self._finalize_node)
        
        workflow.set_entry_point("reason")
        
        workflow.add_conditional_edges(
            "reason",
            lambda state: (
                "finalize" if state.status == ExecutionStatus.FAILED
                else "act" if state.intermediate_outputs.get("next_action", {}).get("type") == "tool_call"
                else "finalize"
            ),
        )
        
        workflow.add_edge("act", "reason")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    # ============================================================================
    # MESSAGE-BASED PROMPTING
    # ============================================================================
    
    def _reason_node(self, state: AgentExecutionState) -> AgentExecutionState:
        """LLM decides next action using message-based prompting."""
        iteration = len(state.reasoning_steps)
        append_chatbot_log(f"=== REASON NODE (iteration {iteration}) ===")
        
        # Check iteration limits
        if not self._should_continue_iteration(state):
            return self._force_finalize(state, "Iteration limit reached")
        
        try:
            # Build message list
            messages = self._build_messages(state)
            
            # Get LLM response with JSON mode
            llm = get_llm(
                tier=self.configuration.llm_tier,
                json_mode=True,
                temperature=0.0
            )
            
            response = llm.invoke(messages)
            decision = self._parse_json_decision(response.content)
            
            append_chatbot_log(f"Decision: {json.dumps(decision, indent=2)}")
            state.add_reasoning_step(json.dumps(decision))
            
            # Validate decision
            if not self._is_decision_valid(state, decision):
                return self._force_finalize(state, "Invalid decision")
            
            self._process_decision(state, decision)
            
        except json.JSONDecodeError as e:
            append_chatbot_log(f"JSON parse error: {e}")
            return self._force_finalize(state, "JSON parsing failed")
            
        except Exception as e:
            append_chatbot_log(f"Error in reason node: {e}")
            state.mark_failed(str(e))
            state.intermediate_outputs["next_action"] = {"type": "finalize"}
        
        return state
    
    def _build_messages(self, state: AgentExecutionState) -> List:
        """
        Build message list for LLM.
        Override this method in subclasses for custom message construction.
        """
        messages = [
            SystemMessage(content=self._get_system_prompt())
        ]
        
        # Add few-shot examples (first iteration only)
        if len(state.reasoning_steps) == 0:
            few_shots = self._get_few_shot_messages()
            if few_shots:
                messages.extend(few_shots)
        
        # Add conversation history
        history_msgs = self._format_conversation_history(state)
        if history_msgs:
            messages.extend(history_msgs)
        
        # Add tool results from previous iterations
        tool_msgs = self._format_tool_results(state)
        if tool_msgs:
            messages.extend(tool_msgs)
        
        # Add current situation/query
        messages.append(
            HumanMessage(content=self._build_current_situation(state))
        )
        
        return messages
    
    def _get_few_shot_messages(self) -> List:
        """
        Get few-shot example messages.
        Override in subclasses to provide agent-specific examples.
        
        Returns:
            List of message objects showing example interactions
        """
        return []  # Default: no examples
    
    def _format_conversation_history(self, state: AgentExecutionState) -> List:
        """Format conversation history as messages."""
        messages = []
        
        if not state.conversation_history:
            return messages
        
        # Only include last few turns to save tokens
        for turn in state.conversation_history[-2:]:
            if 'u' in turn:
                messages.append(HumanMessage(content=turn['u']))
            if 'a' in turn:
                messages.append(AIMessage(content=turn['a']))
        
        return messages
    
    def _format_tool_results(self, state: AgentExecutionState) -> List:
        """Format recent tool executions as messages."""
        messages = []
        
        if not state.tool_executions:
            return messages
        
        # Build summary of recent tool calls
        parts = ["PREVIOUS TOOL RESULTS:"]
        
        for i, exec in enumerate(state.tool_executions[-3:], 1):
            status = "✓ SUCCESS" if exec.succeeded else "✗ FAILED"
            parts.append(f"\n{i}. {exec.tool_name} ({status}) [{exec.execution_time_ms}ms]")
            
            if exec.error:
                parts.append(f"   Error: {exec.error}")
            else:
                result_preview = str(exec.result)[:300]
                if len(str(exec.result)) > 300:
                    result_preview += "..."
                parts.append(f"   Result: {result_preview}")
        
        messages.append(HumanMessage(content="\n".join(parts)))
        
        return messages
    
    def _build_current_situation(self, state: AgentExecutionState) -> str:
        """
        Build description of current situation.
        Override in subclasses for agent-specific context.
        """
        parts = []
        
        # Available tools
        parts.append("AVAILABLE TOOLS:")
        if self.tool_executor._tools:
            for tool_name, tool in self.tool_executor._tools.items():
                sig = tool.get_signature()
                params = []
                for pname, param in sig.parameters.items():
                    if pname in ('self', 'cls'):
                        continue
                    param_type = getattr(param.annotation, '__name__', 'any')
                    is_required = param.default == param.empty
                    params.append(f"{pname}: {param_type}" + ("" if is_required else " (optional)"))
                
                param_str = ", ".join(params) if params else "no parameters"
                parts.append(f"- {tool_name}({param_str}): {tool.description}")
        else:
            parts.append("- No tools available")
        
        parts.append("\n" + "="*60)
        
        # Current query
        parts.append(f"\nUSER QUERY:\n{state.input_text}")
        
        # Progress indicators
        book_ids = state.intermediate_outputs.get("book_ids", [])
        if book_ids:
            parts.append(f"\nBooks found so far: {len(book_ids)} ({book_ids[:5]}...)")
        
        parts.append("\n" + "="*60)
        
        # JSON format instructions
        parts.append("\nDECISION FORMAT:")
        parts.append("Respond with valid JSON in ONE of these formats:\n")
        
        parts.append("1. To call a tool:")
        parts.append(json.dumps({
            "action": "tool_call",
            "tool": "tool_name",
            "arguments": {"param1": "value1"},
            "reasoning": "brief explanation"
        }, indent=2))
        
        parts.append("\n2. To provide final answer:")
        parts.append(json.dumps({
            "action": "answer",
            "text": "your response to the user",
            "reasoning": "brief explanation"
        }, indent=2))
        
        parts.append("\nRULES:")
        parts.append("- Return ONLY valid JSON")
        parts.append("- 'action' must be 'tool_call' or 'answer'")
        parts.append("- Arguments must be a dict matching tool parameters")
        parts.append("- Don't call the same tool twice with identical arguments")
        parts.append("- If you have sufficient information, choose 'answer'")
        
        return "\n".join(parts)
    
    # ============================================================================
    # DECISION PROCESSING
    # ============================================================================
    
    def _parse_json_decision(self, content: str) -> Dict[str, Any]:
        """Parse and validate JSON decision from LLM."""
        content = content.strip()
        
        # Remove code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            if content.startswith("json"):
                content = content[4:].strip()
        
        decision = json.loads(content)
        
        if not isinstance(decision, dict):
            raise ValueError("Decision must be a JSON object")
        
        if "action" not in decision:
            raise ValueError("Decision missing required 'action' field")
        
        action = decision["action"]
        
        if action == "tool_call":
            if "tool" not in decision:
                raise ValueError("tool_call decision missing 'tool' field")
            if "arguments" not in decision:
                decision["arguments"] = {}
            if not isinstance(decision["arguments"], dict):
                raise ValueError("'arguments' must be a JSON object/dict")
        
        elif action == "answer":
            if "text" not in decision:
                raise ValueError("answer decision missing 'text' field")
            if not isinstance(decision["text"], str):
                raise ValueError("'text' must be a string")
        
        else:
            raise ValueError(f"Invalid action '{action}'. Must be 'tool_call' or 'answer'")
        
        return decision
    
    def _is_decision_valid(self, state: AgentExecutionState, decision: Dict[str, Any]) -> bool:
        """Validate decision makes sense given current state."""
        action = decision.get("action")
        
        if action == "tool_call":
            tool_name = decision.get("tool")
            
            if tool_name not in self.tool_executor._tools:
                append_chatbot_log(f"Tool '{tool_name}' not available")
                return False
        
        return True
    
    def _process_decision(self, state: AgentExecutionState, decision: Dict[str, Any]) -> None:
        """Process parsed decision and update state."""
        action = decision["action"]
        
        if action == "tool_call":
            state.intermediate_outputs["next_action"] = {
                "type": "tool_call",
                "tool": decision["tool"],
                "args": decision["arguments"]
            }
            
            if "reasoning" in decision:
                append_chatbot_log(f"Reasoning: {decision['reasoning']}")
        
        elif action == "answer":
            state.intermediate_outputs["final_answer"] = decision["text"]
            state.intermediate_outputs["next_action"] = {"type": "finalize"}
            
            if "reasoning" in decision:
                append_chatbot_log(f"Reasoning: {decision['reasoning']}")
    
    # ============================================================================
    # ITERATION CONTROL
    # ============================================================================
    
    def _should_continue_iteration(self, state: AgentExecutionState) -> bool:
        """Check if agent should continue iterating."""
        if len(state.reasoning_steps) >= self.configuration.max_iterations:
            append_chatbot_log("Max iterations reached")
            return False
        
        elapsed = time.time() - state.start_time
        if elapsed > self.configuration.timeout_seconds:
            append_chatbot_log(f"Timeout reached ({elapsed:.1f}s)")
            return False
        
        if self._is_stuck_in_loop(state):
            append_chatbot_log("Detected stuck loop")
            return False
        
        return True
    
    def _is_stuck_in_loop(self, state: AgentExecutionState) -> bool:
        """Detect if agent is calling same tool repeatedly."""
        if len(state.tool_executions) < 3:
            return False
        
        recent = state.tool_executions[-3:]
        
        if len(set(e.tool_name for e in recent)) == 1:
            args_strs = [json.dumps(e.arguments, sort_keys=True) for e in recent]
            if len(set(args_strs)) == 1:
                if all(not e.succeeded for e in recent):
                    return True
                results = [str(e.result) for e in recent if e.succeeded]
                if len(set(results)) == 1 and len(results) == len(recent):
                    return True
        
        return False
    
    def _force_finalize(self, state: AgentExecutionState, reason: str) -> AgentExecutionState:
        """Force finalization with synthesized answer."""
        append_chatbot_log(f"Forcing finalization: {reason}")
        
        if state.tool_executions:
            answer = self._synthesize_answer_from_tools(state)
        else:
            answer = "I wasn't able to complete that request. Please try rephrasing your question."
        
        state.intermediate_outputs["final_answer"] = answer
        state.intermediate_outputs["next_action"] = {"type": "finalize"}
        
        return state
    
    def _synthesize_answer_from_tools(self, state: AgentExecutionState) -> str:
        """Quick synthesis when forcing finalization."""
        try:
            llm = get_llm(tier="small", temperature=0.1, timeout=10)
            
            results_summary = []
            for exec in state.tool_executions[-3:]:
                if exec.succeeded:
                    result_str = str(exec.result)[:200]
                    results_summary.append(f"- {exec.tool_name}: {result_str}")
            
            if not results_summary:
                return "I attempted to gather information but encountered issues."
            
            prompt = (
                f"User asked: {state.input_text}\n\n"
                f"Tool results:\n" + "\n".join(results_summary) + "\n\n"
                f"Provide a brief, helpful response (2-3 sentences)."
            )
            
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            append_chatbot_log(f"Synthesis failed: {e}")
            return "I found some information but had trouble summarizing it."
    
    # ============================================================================
    # ACT AND FINALIZE NODES
    # ============================================================================
    
    def _act_node(self, state: AgentExecutionState) -> AgentExecutionState:
        """Execute the tool chosen by LLM."""
        append_chatbot_log("=== ACT NODE ===")
        
        try:
            action = state.intermediate_outputs.get("next_action", {})
            tool_name = action.get("tool", "")
            tool_args = action.get("args", {})
            
            if not tool_name:
                append_chatbot_log("No tool specified in action")
                return state
            
            append_chatbot_log(f"Executing: {tool_name}")
            append_chatbot_log(f"Arguments: {json.dumps(tool_args, indent=2)}")
            
            tool_exec = self.tool_executor.execute(tool_name, tool_args)
            state.add_tool_execution(tool_exec)
            
            append_chatbot_log(
                f"Result: {'✓ success' if tool_exec.succeeded else '✗ failed'} "
                f"({tool_exec.execution_time_ms}ms)"
            )
            
            # Capture book IDs if this is a recommendation tool
            if tool_exec.succeeded and self.tool_executor.is_book_recommendation_tool(tool_name):
                book_ids = self.tool_executor.extract_book_ids_from_result(tool_exec.result)
                
                if book_ids:
                    existing = state.intermediate_outputs.get("book_ids", [])
                    seen = set(existing)
                    new_count = 0
                    
                    for bid in book_ids:
                        if bid not in seen:
                            existing.append(bid)
                            seen.add(bid)
                            new_count += 1
                    
                    state.intermediate_outputs["book_ids"] = existing
                    append_chatbot_log(
                        f"Captured {new_count} new book IDs "
                        f"(total: {len(existing)})"
                    )
        
        except Exception as e:
            append_chatbot_log(f"Error in act node: {e}")
        
        return state
    
    def _finalize_node(self, state: AgentExecutionState) -> AgentExecutionState:
        """Mark execution as complete."""
        append_chatbot_log("=== FINALIZE NODE ===")
        
        if "final_answer" not in state.intermediate_outputs:
            append_chatbot_log("WARNING: No final_answer - generating fallback")
            state.intermediate_outputs["final_answer"] = (
                "I apologize, I couldn't generate a proper response."
            )
        
        state.mark_completed()
        
        append_chatbot_log(
            f"Finalized: {len(state.tool_executions)} tools, "
            f"{len(state.intermediate_outputs.get('book_ids', []))} books, "
            f"{state.execution_time_ms}ms"
        )
        
        return state
    
    # ============================================================================
    # ABSTRACT METHODS (subclasses must implement)
    # ============================================================================
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get system prompt for this agent type."""
        pass
    
    @abstractmethod
    def _get_target_category(self) -> str:
        """Get target category for this agent."""
        pass
    
    # ============================================================================
    # MAIN EXECUTION
    # ============================================================================
    
    def execute(self, request: AgentRequest) -> AgentResponse:
        """Main execution method - implements BaseAgent interface."""
        append_chatbot_log(
            f"\n{'='*60}\n"
            f"{self.__class__.__name__.upper()} EXECUTION START\n"
            f"Query: {request.user_text[:100]}...\n"
            f"{'='*60}"
        )
        
        state = AgentExecutionState(
            status=ExecutionStatus.RUNNING,
            input_text=request.user_text,
            conversation_history=request.conversation_history,
        )
        
        try:
            result = run_with_timeout(
                self.graph.invoke,
                self.configuration.timeout_seconds,
                state
            )
            
            if isinstance(result, dict):
                final_state = AgentExecutionState(
                    status=ExecutionStatus(result.get("status", "completed")),
                    input_text=result.get("input_text", ""),
                    conversation_history=result.get("conversation_history", []),
                    tool_executions=result.get("tool_executions", []),
                    reasoning_steps=result.get("reasoning_steps", []),
                    intermediate_outputs=result.get("intermediate_outputs", {}),
                    error_message=result.get("error_message"),
                    start_time=result.get("start_time", time.time()),
                    end_time=result.get("end_time"),
                )
            else:
                final_state = result
            
            books = self.result_processor.extract_book_recommendations(final_state)
            text = self.result_processor.format_response_text(final_state)
            citations = self.result_processor.extract_citations(final_state)
            
            response = AgentResponse(
                text=text,
                target_category=self._get_target_category(),
                success=final_state.status == ExecutionStatus.COMPLETED,
                book_recommendations=books,
                citations=citations,
                execution_state=final_state,
                policy_version=self.configuration.policy_name,
            )
            
            append_chatbot_log(
                f"\n{'='*60}\n"
                f"EXECUTION COMPLETE\n"
                f"Books: {len(books)}, Tools: {len(final_state.tool_executions)}, "
                f"Time: {final_state.execution_time_ms}ms\n"
                f"{'='*60}\n"
            )
            
            return response
            
        except TimeoutException as e:
            append_chatbot_log(f"\n[HARD TIMEOUT] {str(e)}\n")
            
            final_answer = self._synthesize_answer_from_tools(state) if state.tool_executions else \
                        "I'm taking too long to respond. Please try rephrasing your request."
            
            return AgentResponse(
                text=final_answer,
                target_category=self._get_target_category(),
                success=False,
                execution_state=state,
                policy_version=self.configuration.policy_name,
            )

        except Exception as e:
            append_chatbot_log(f"\n[EXECUTION ERROR] {str(e)}\n")
            
            return AgentResponse(
                text=f"I encountered an error: {str(e)}",
                target_category=self._get_target_category(),
                success=False,
                policy_version=self.configuration.policy_name,
            )