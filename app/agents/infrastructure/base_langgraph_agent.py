# app/agents/infrastructure/base_langgraph_agent.py
"""
Base LangGraph agent implementing ReAct pattern with clean state management.
LLM drives tool selection; state captures results during execution.
"""
from typing import Dict, Any, Literal, Optional
from abc import abstractmethod

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage

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


class BaseLangGraphAgent(BaseAgent):
    """
    Base agent implementing ReAct loop with LangGraph.
    
    Pattern:
        entry → reason (LLM decides) → should_continue?
                                           ↓ continue    ↓ finalize
                                          act         finalize → END
                                           ↓
                                        reason (loop)
    """
    
    def __init__(self, configuration, ctx_user=None, ctx_db=None):
        super().__init__(configuration)
        
        # LLM for reasoning
        self.llm = get_llm(tier=configuration.llm_tier)
        
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
        """Create tool registry based on agent capabilities. Override in subclasses."""
        config = self.configuration
        
        gates = InternalToolGates(
            user_num_ratings=0,  # Will be set per request
            warm_threshold=10,
            profile_allowed=False  # Will be set per request
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
        """Build the ReAct loop graph."""
        workflow = StateGraph(AgentExecutionState)
        
        # Core ReAct nodes
        workflow.add_node("reason", self._reason_node)
        workflow.add_node("act", self._act_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Flow
        workflow.set_entry_point("reason")
        workflow.add_conditional_edges(
            "reason",
            self._should_continue,
            {
                "continue": "act",
                "finalize": "finalize"
            }
        )
        workflow.add_edge("act", "reason")  # Loop back
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _reason_node(self, state: AgentExecutionState) -> AgentExecutionState:
        """
        LLM decides what to do next: call a tool or provide final answer.
        """
        append_chatbot_log(f"=== REASON NODE (iteration {len(state.reasoning_steps)}) ===")
        
        # Check iteration limit
        if len(state.reasoning_steps) >= 10:
            state.intermediate_outputs["next_action"] = {"type": "finalize"}
            state.add_reasoning_step("Max iterations reached - finalizing")
            return state
        
        try:
            # Build prompt for LLM
            prompt_parts = []
            
            # System instructions
            prompt_parts.append(self._get_system_prompt())
            
            # Available tools
            prompt_parts.append("\n" + self._format_available_tools())
            
            # User query
            prompt_parts.append(f"\nUser Query: {state.input_text}")
            
            # Previous tool results (if any)
            if state.tool_executions:
                prompt_parts.append("\nPrevious Tool Results:")
                for exec in state.tool_executions[-3:]:  # Last 3 tools
                    result_str = str(exec.result)[:500]
                    prompt_parts.append(
                        f"- {exec.tool_name}: {result_str}..."
                    )
            
            # Instructions for next action
            prompt_parts.append(
                "\nDecide your next action. Respond with ONE of:\n"
                "1. TOOL: <tool_name> <arguments> - to call a tool\n"
                "2. ANSWER: <final response> - when ready to respond to user\n"
            )
            
            full_prompt = "\n".join(prompt_parts)
            
            # Get LLM decision
            messages = [HumanMessage(content=full_prompt)]
            response = self.llm.invoke(messages)
            
            decision = response.content if hasattr(response, 'content') else str(response)
            state.add_reasoning_step(decision)
            
            append_chatbot_log(f"LLM Decision: {decision[:200]}...")
            
            # Parse decision
            self._parse_llm_decision(state, decision)
            
        except Exception as e:
            append_chatbot_log(f"Error in reason node: {e}")
            state.mark_failed(str(e))
            state.intermediate_outputs["next_action"] = {"type": "finalize"}
        
        return state
    
    def _parse_llm_decision(self, state: AgentExecutionState, decision: str) -> None:
        """Parse LLM's decision and store in state."""
        decision_upper = decision.upper().strip()
        
        if decision_upper.startswith("TOOL:"):
            # Extract tool call
            tool_line = decision[5:].strip()
            parts = tool_line.split(None, 1)
            
            if parts:
                tool_name = parts[0].strip()
                tool_args = parts[1] if len(parts) > 1 else ""
                
                state.intermediate_outputs["next_action"] = {
                    "type": "tool_call",
                    "tool": tool_name,
                    "args": tool_args
                }
                append_chatbot_log(f"Parsed tool call: {tool_name}")
            else:
                # Malformed tool call - finalize
                state.intermediate_outputs["next_action"] = {"type": "finalize"}
        
        elif decision_upper.startswith("ANSWER:"):
            # LLM ready to provide final answer
            answer_text = decision[7:].strip()
            state.intermediate_outputs["final_answer"] = answer_text
            state.intermediate_outputs["next_action"] = {"type": "finalize"}
        
        else:
            # Unclear - treat as final answer
            state.intermediate_outputs["final_answer"] = decision
            state.intermediate_outputs["next_action"] = {"type": "finalize"}
    
    def _should_continue(self, state: AgentExecutionState) -> Literal["continue", "finalize"]:
        """Routing: continue with tool execution or finalize?"""
        if state.status == ExecutionStatus.FAILED:
            return "finalize"
        
        action = state.intermediate_outputs.get("next_action", {})
        
        if action.get("type") == "tool_call":
            return "continue"
        
        return "finalize"
    
    def _act_node(self, state: AgentExecutionState) -> AgentExecutionState:
        """Execute the tool chosen by LLM and capture results in state."""
        append_chatbot_log("=== ACT NODE ===")
        
        try:
            action = state.intermediate_outputs.get("next_action", {})
            tool_name = action.get("tool", "")
            tool_args_str = action.get("args", "")
            
            # Convert string args to dict for tool executor
            args_dict = {"input": tool_args_str}
            
            # Execute tool
            tool_exec = self.tool_executor.execute(tool_name, args_dict)
            state.add_tool_execution(tool_exec)
            
            append_chatbot_log(
                f"Executed {tool_name}: "
                f"{'success' if tool_exec.succeeded else 'failed'}"
            )
            
            # 🔑 KEY: Capture book IDs immediately if this tool returns them
            if tool_exec.succeeded and self.tool_executor.is_book_recommendation_tool(tool_name):
                book_ids = self.tool_executor.extract_book_ids_from_result(tool_exec.result)
                
                if book_ids:
                    # Accumulate book IDs (don't replace)
                    existing = state.intermediate_outputs.get("book_ids", [])
                    # Dedupe while preserving order
                    seen = set(existing)
                    for bid in book_ids:
                        if bid not in seen:
                            existing.append(bid)
                            seen.add(bid)
                    
                    state.intermediate_outputs["book_ids"] = existing
                    append_chatbot_log(f"Captured {len(book_ids)} book IDs")
        
        except Exception as e:
            append_chatbot_log(f"Error in act node: {e}")
            # Don't fail - let LLM decide what to do next
        
        return state
    
    def _finalize_node(self, state: AgentExecutionState) -> AgentExecutionState:
        """Generate final response if needed."""
        append_chatbot_log("=== FINALIZE NODE ===")
        
        # Check if we already have a final answer from reason node
        if "final_answer" in state.intermediate_outputs:
            state.mark_completed()
            return state
        
        try:
            # Generate final answer based on tool results
            context_parts = [self._get_system_prompt()]
            context_parts.append(f"\nUser Query: {state.input_text}")
            
            if state.tool_executions:
                context_parts.append("\nTool Results:")
                for exec in state.tool_executions:
                    result_str = str(exec.result)[:500]
                    context_parts.append(f"- {exec.tool_name}: {result_str}")
            
            context_parts.append(
                "\nNow provide a natural, helpful response to the user based on "
                "the tool results above."
            )
            
            full_context = "\n".join(context_parts)
            messages = [HumanMessage(content=full_context)]
            response = self.llm.invoke(messages)
            
            final_text = response.content if hasattr(response, 'content') else str(response)
            state.intermediate_outputs["final_answer"] = final_text
            state.mark_completed()
            
        except Exception as e:
            append_chatbot_log(f"Error in finalize node: {e}")
            state.intermediate_outputs["final_answer"] = (
                "I encountered an error generating the response."
            )
            state.mark_failed(str(e))
        
        return state
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get system prompt for this agent type. Override in subclasses."""
        pass
    
    def _format_available_tools(self) -> str:
        """Format available tools for LLM prompt."""
        if not self.tool_executor._tools:
            return "Available Tools: None"
        
        lines = ["Available Tools:"]
        for tool_name, tool in self.tool_executor._tools.items():
            desc = getattr(tool, 'description', 'No description')
            lines.append(f"- {tool_name}: {desc}")
        
        return "\n".join(lines)
    
    def execute(self, request: AgentRequest) -> AgentResponse:
        """
        Main execution method - implements BaseAgent interface.
        
        Flow:
            1. Initialize state from request
            2. Run LangGraph workflow
            3. Extract results using StandardResultProcessor
            4. Return AgentResponse
        """
        append_chatbot_log(
            f"\n=== {self.__class__.__name__.upper()} EXECUTION START ==="
        )
        
        # Initialize execution state
        state = AgentExecutionState(
            status=ExecutionStatus.RUNNING,
            input_text=request.user_text,
            conversation_history=request.conversation_history,
        )
        
        try:
            # Run the graph
            final_state = self.graph.invoke(state)
            
            # Extract results using domain processor
            books = self.result_processor.extract_book_recommendations(final_state)
            text = self.result_processor.format_response_text(final_state)
            citations = self.result_processor.extract_citations(final_state)
            
            # Build response
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
                f"Execution completed: {len(books)} books, "
                f"{len(final_state.tool_executions)} tools used"
            )
            append_chatbot_log(f"=== EXECUTION END ===\n")
            
            return response
            
        except Exception as e:
            append_chatbot_log(f"[EXECUTION ERROR] {str(e)}")
            
            # Return error response
            return AgentResponse(
                text=f"I encountered an error: {str(e)}",
                target_category=self._get_target_category(),
                success=False,
                policy_version=self.configuration.policy_name,
            )
    
    @abstractmethod
    def _get_target_category(self) -> str:
        """Get target category for this agent. Override in subclasses."""
        pass
