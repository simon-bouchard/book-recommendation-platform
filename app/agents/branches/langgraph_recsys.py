# app/agents/branches/langgraph_recsys.py

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from typing import Dict, List, Any, Optional, Literal
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import Tool
import json
import re

from app.agents.tools.registry import InternalToolGates
from app.agents.schemas import AgentResult, ToolCall
from app.agents.langgraph_base import LangGraphAgent, AgentState
from app.agents.logging import append_chatbot_log

class RecsysAgentState(TypedDict):
    """Extended state for the recommendation workflow."""
    user_input: str
    conversation_history: List[dict]
    system_prompt: str
    tools_used: List[dict]
    tool_results: Dict[str, Any]
    reasoning: str
    final_answer: str
    book_ids: Optional[List[int]]
    error: Optional[str]
    messages: List[Any]  # Track conversation messages for tool calling
    iteration_count: int
    max_iterations: int

class RecsysLangGraphAgent(LangGraphAgent):
    """LangGraph-based recommendation agent with proper ReAct-style tool calling."""
    
    def __init__(self, current_user=None, db=None, user_num_ratings=None, **kwargs):
        gates = InternalToolGates(
            user_num_ratings=user_num_ratings,
            warm_threshold=10,
            profile_allowed=kwargs.get('allow_profile', False)
        )
        
        super().__init__(
            policy_name="recsys.system.md",
            web=False, docs=False, internal=True,
            gates=gates, ctx_user=current_user, ctx_db=db,
            llm_tier="large",
            allowed_names={
                "als_recs", "subject_hybrid_pool", "subject_id_search",
                "book_semantic_search", "return_book_ids", "user-profile", "recent-interactions"
            }
        )
        
        # Create tool descriptions for the LLM
        self.tool_descriptions = self._build_tool_descriptions()
        append_chatbot_log(f"RecsysLangGraphAgent initialized with {len(self.tools)} tools")
    
    def _build_graph(self) -> StateGraph:
        """Define the improved recommendation workflow."""
        workflow = StateGraph(RecsysAgentState)
        
        workflow.add_node("reason_and_act", self._reason_and_act)
        workflow.add_node("execute_tool", self._execute_tool)  
        workflow.add_node("finalize_response", self._finalize_response)
        
        workflow.set_entry_point("reason_and_act")
        
        # Conditional routing based on LLM's decision
        workflow.add_conditional_edges(
            "reason_and_act",
            self._should_continue,
            {
                "continue": "execute_tool",
                "finalize": "finalize_response"
            }
        )
        
        workflow.add_edge("execute_tool", "reason_and_act")  # Loop back for more reasoning
        workflow.add_edge("finalize_response", END)
        
        return workflow.compile()
    
    def _reason_and_act(self, state: RecsysAgentState) -> RecsysAgentState:
        """Main ReAct node where LLM decides what to do next."""
        append_chatbot_log(f"=== REASON_AND_ACT (iteration {state.get('iteration_count', 0)}) ===")
        
        # Initialize iteration tracking
        if "iteration_count" not in state:
            state["iteration_count"] = 0
            state["max_iterations"] = 5
        
        # Check iteration limit
        if state["iteration_count"] >= state["max_iterations"]:
            state["reasoning"] = "Maximum iterations reached"
            return state
        
        state["iteration_count"] = state["iteration_count"] + 1
        
        try:
            # Build context for LLM decision
            context_parts = []
            context_parts.append(state["system_prompt"])
            context_parts.append(f"\nUser Request: {state['user_input']}")
            
            # Add tool results from previous iterations
            if state["tools_used"]:
                context_parts.append("\nPrevious Tool Results:")
                for tool_use in state["tools_used"]:
                    context_parts.append(f"- {tool_use['name']}: {str(tool_use['result'])[:200]}...")
            
            # Add available tools
            context_parts.append(f"\nAvailable Tools: {self.tool_descriptions}")
            
            # Ask LLM to decide next action
            context_parts.append("\nWhat should I do next? Respond with either:")
            context_parts.append("1. TOOL: <tool_name> <arguments> - to call a specific tool")
            context_parts.append("2. FINALIZE - to generate the final recommendation")
            
            full_context = "\n".join(context_parts)
            
            messages = [HumanMessage(content=full_context)]
            response = self.llm.invoke(messages)
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            state["reasoning"] = response_text
            append_chatbot_log(f"LLM Decision: {response_text}")
            
            # Parse the response to determine action
            if response_text.upper().startswith("TOOL:"):
                # Extract tool call
                tool_line = response_text[5:].strip()  # Remove "TOOL:"
                parts = tool_line.split(None, 1)  # Split into tool_name and args
                if parts:
                    tool_name = parts[0]
                    tool_args = parts[1] if len(parts) > 1 else ""
                    state["pending_tool_calls"] = [{"name": tool_name, "args": tool_args}]
                    append_chatbot_log(f"Parsed tool call: {tool_name} with args: {tool_args}")
            else:
                # No tool call or ready to finalize
                state["pending_tool_calls"] = []
                append_chatbot_log("No tool call or ready to finalize")
                
        except Exception as e:
            append_chatbot_log(f"Error in reason_and_act: {e}")
            state["error"] = str(e)
            
        return state
    
    def _should_continue(self, state: RecsysAgentState) -> Literal["continue", "finalize"]:
        """Decide whether to continue with tool execution or finalize."""
        # If there's an error, finalize
        if state.get("error"):
            return "finalize"
            
        # If we have pending tool calls, continue
        if state.get("pending_tool_calls"):
            return "continue"
            
        # If we've reached max iterations, finalize
        if state.get("iteration_count", 0) >= state.get("max_iterations", 5):
            return "finalize"
            
        # Otherwise finalize (no more work to do)
        return "finalize"
    
    def _execute_tool(self, state: RecsysAgentState) -> RecsysAgentState:
        """Execute the tools requested by the LLM."""
        append_chatbot_log("=== EXECUTE_TOOL ===")
        
        tool_calls = state.get("pending_tool_calls", [])
        if not tool_calls:
            return state
        
        # Execute each tool call
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", "")
            
            append_chatbot_log(f"Executing tool: {tool_name} with args: {tool_args}")
            
            try:
                if tool_name in self.tools:
                    # Execute the tool with string args (as per your existing pattern)
                    result = self.tools[tool_name].func(tool_args)
                    
                    # Store in tools_used for final result
                    state["tools_used"].append({
                        "name": tool_name,
                        "args": {"input": tool_args},
                        "result": result
                    })
                    
                    append_chatbot_log(f"Tool {tool_name} result: {str(result)[:200]}...")
                    
                else:
                    append_chatbot_log(f"Unknown tool: {tool_name}")
                    
            except Exception as e:
                error_msg = f"Error executing {tool_name}: {str(e)}"
                append_chatbot_log(error_msg)
        
        # Clear pending tool calls
        state["pending_tool_calls"] = []
        return state
    
    def _build_tool_descriptions(self) -> str:
        """Build a simple description of available tools for the LLM."""
        descriptions = []
        for tool_name, tool in self.tools.items():
            desc = getattr(tool, 'description', 'No description')
            descriptions.append(f"- {tool_name}: {desc}")
        return "\n".join(descriptions)
    
    def _finalize_response(self, state: RecsysAgentState) -> RecsysAgentState:
        """Generate final recommendation response and extract book IDs."""
        append_chatbot_log("=== FINALIZE_RESPONSE ===")
        
        if state.get("error"):
            state["final_answer"] = "I encountered an error processing your request."
            return state
        
        try:
            # Build context for final response generation
            context_parts = []
            context_parts.append(state["system_prompt"])
            context_parts.append(f"\nUser Request: {state['user_input']}")
            
            # Add all tool results
            if state["tools_used"]:
                context_parts.append("\nTool Results:")
                for tool_use in state["tools_used"]:
                    context_parts.append(f"- {tool_use['name']}: {str(tool_use['result'])[:500]}...")
            
            context_parts.append("\nNow generate a natural language recommendation response based on the tool results.")
            context_parts.append("Focus on books that best match the user's request.")
            
            full_context = "\n".join(context_parts)
            
            messages = [HumanMessage(content=full_context)]
            response = self.llm.invoke(messages)
            
            # Extract the natural language response
            response_text = response.content if hasattr(response, 'content') else str(response)
            state["final_answer"] = response_text
            
            # Ensure we have book IDs and call return_book_ids properly
            book_ids = self._extract_book_ids_from_tools(state["tools_used"])
            
            if book_ids and "return_book_ids" in self.tools:
                try:
                    # Call return_book_ids with the extracted IDs to ensure proper formatting
                    result = self.tools["return_book_ids"].func(json.dumps(book_ids))
                    
                    # Store this as the final tool call for runtime extraction
                    final_tool_call = {
                        "name": "return_book_ids",
                        "args": {"input": book_ids},
                        "result": result
                    }
                    
                    # Replace any existing return_book_ids call or add new one
                    filtered_tools = [t for t in state["tools_used"] if t.get("name") != "return_book_ids"]
                    filtered_tools.append(final_tool_call)
                    state["tools_used"] = filtered_tools
                    
                    # Also store in state for our own reference
                    if isinstance(result, str):
                        try:
                            result_data = json.loads(result)
                            if isinstance(result_data, dict) and "book_ids" in result_data:
                                state["book_ids"] = result_data["book_ids"]
                        except json.JSONDecodeError:
                            state["book_ids"] = book_ids
                    else:
                        state["book_ids"] = book_ids
                        
                    append_chatbot_log(f"Final book IDs: {state.get('book_ids')}")
                    
                except Exception as e:
                    append_chatbot_log(f"Error calling return_book_ids: {e}")
                    # Fallback: ensure we still have the IDs
                    state["book_ids"] = book_ids
            elif book_ids:
                # No return_book_ids tool available, but we have IDs
                state["book_ids"] = book_ids
                append_chatbot_log(f"Book IDs extracted without return_book_ids tool: {book_ids}")
            else:
                append_chatbot_log("No book IDs found in tool results")
                
        except Exception as e:
            append_chatbot_log(f"Error in finalize_response: {e}")
            state["error"] = str(e)
            state["final_answer"] = "I encountered an error generating the final response."
        
        return state
    
    def _extract_book_ids_from_tools(self, tools_used: List[dict]) -> List[int]:
        """Extract book IDs from tool results, matching runtime.py extraction logic."""
        book_ids = []
        
        # Look for return_book_ids calls first (most authoritative)
        for tool in reversed(tools_used):
            if tool.get("name") == "return_book_ids":
                try:
                    result = tool.get("result", "")
                    if isinstance(result, str):
                        data = json.loads(result)
                        if isinstance(data, dict) and "book_ids" in data:
                            return [int(x) for x in data["book_ids"] if str(x).strip()]
                except:
                    pass
        
        # Fallback: extract item_idx from other tool results  
        # This matches the logic in runtime.py extract_book_ids_from_steps
        for tool in tools_used:
            try:
                result = tool.get("result", "")
                if isinstance(result, str) and result.strip():
                    # Try to parse as JSON list of objects with item_idx
                    if result.strip().startswith("["):
                        data = json.loads(result)
                        if isinstance(data, list):
                            for item in data[:12]:  # Limit to reasonable number
                                if isinstance(item, dict) and "item_idx" in item:
                                    book_ids.append(int(item["item_idx"]))
                            if book_ids:  # Found some IDs, stop looking
                                break
            except:
                continue
        
        # Deduplicate while preserving order (like runtime.py does)
        seen = set()
        deduped = []
        for id in book_ids:
            if id not in seen:
                seen.add(id)
                deduped.append(id)
        
        return deduped
    
    def run(self, inp) -> AgentResult:
        """Override to initialize state properly and set correct target."""
        # Initialize extended state
        if isinstance(inp, str):
            user_input = inp
            history = []
        else:
            user_input = inp.user_text
            history = inp.full_history
        
        initial_state = RecsysAgentState(
            user_input=user_input,
            conversation_history=history,
            system_prompt=self.system_prompt,
            tools_used=[],
            tool_results={},
            reasoning="",
            final_answer="",
            book_ids=None,
            error=None,
            messages=[],
            iteration_count=0,
            max_iterations=5
        )
        
        try:
            # Execute the graph
            final_state = self.graph.invoke(initial_state)
            
            return AgentResult(
                target="recsys",
                text=final_state.get("final_answer", ""),
                success=not final_state.get("error"),
                tool_calls=self._format_tool_calls(final_state.get("tools_used", [])),
                book_ids=final_state.get("book_ids"),
                policy_version=self.policy_name,
            )
            
        except Exception as e:
            append_chatbot_log(f"[AGENT_ERROR] {str(e)}")
            return AgentResult(
                target="recsys",
                text=f"I encountered an error: {str(e)}",
                success=False,
                tool_calls=[],
                policy_version=self.policy_name,
            )