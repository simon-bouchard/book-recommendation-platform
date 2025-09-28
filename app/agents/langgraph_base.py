# app/agents/langgraph_base.py
# Complete fixed version with proper debug output

from typing import Dict, List, Any, Optional, Set, Union
from typing_extensions import TypedDict
import json

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.agents.prompts.loader import read_prompt
from app.agents.settings import get_llm
from app.agents.tools.registry import ToolRegistry, InternalToolGates
from app.agents.schemas import AgentResult, TurnInput, ToolCall
from app.agents.logging import append_chatbot_log


class AgentState(TypedDict):
    """Shared state structure for all LangGraph agents."""
    user_input: str
    conversation_history: List[dict]
    system_prompt: str
    tools_used: List[dict]
    tool_results: Dict[str, Any]
    reasoning: str
    final_answer: str
    book_ids: Optional[List[int]]
    error: Optional[str]


class LangGraphAgent:
    """Base class for LangGraph-based agents."""
    
    def __init__(
        self,
        *,
        policy_name: str,
        web: bool = False,
        docs: bool = False,
        internal: bool = False,
        gates: Optional[InternalToolGates] = None,
        ctx_user: Any = None,
        ctx_db: Any = None,
        llm_tier: Optional[str] = None,
        allowed_names: Optional[Set[str]] = None,
        docs_manifest: bool = False,
    ):
        self.policy_name = policy_name
        
        # Create LLM with debug output
        self.llm = get_llm(tier=llm_tier)
        
        # Try different ways to access model info
        model_attr = getattr(self.llm, 'model', 'not_found')
        base_url_attr = getattr(self.llm, 'base_url', 'not_found') 
        api_key_attr = getattr(self.llm, 'api_key', 'not_found')
        
        # Build system prompt same as before
        persona = read_prompt("persona.system.md")
        policy = read_prompt(policy_name)
        final_answer = read_prompt("final_answer.system.md")
        
        if docs_manifest:
            # Handle docs manifest injection if needed
            pass
            
        self.system_prompt = f"{persona}\n\n{policy}\n\n{final_answer}".strip()
        
        # Setup tools (reuse existing registry)
        self.registry = ToolRegistry(
            web=web, docs=docs, internal=internal, gates=gates or InternalToolGates(),
            ctx_user=ctx_user, ctx_db=ctx_db
        )
        self.tools = {
            t.name: t for t in self.registry.get_tools() 
            if not allowed_names or t.name in allowed_names
        }
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Override this in subclasses to define specific workflows."""
        workflow = StateGraph(AgentState)
        
        # Default simple workflow
        workflow.add_node("process", self._process_request)
        workflow.set_entry_point("process")
        workflow.add_edge("process", END)
        
        return workflow.compile()
    
    def _process_request(self, state: AgentState) -> AgentState:
        """Default processing - just call LLM with system prompt."""
        messages = [
            SystemMessage(content=state["system_prompt"]),
            HumanMessage(content=state["user_input"])
        ]
        
        response = self.llm.invoke(messages)
        state["final_answer"] = response.content
        return state
    
    def run(self, inp: Union[str, TurnInput]) -> AgentResult:
        """Main entry point - same interface as BaseLLMAgent."""
        if isinstance(inp, str):
            user_input = inp
            history = []
        else:
            user_input = inp.user_text
            history = inp.full_history
        
        # Initialize state
        initial_state = AgentState(
            user_input=user_input,
            conversation_history=history,
            system_prompt=self.system_prompt,
            tools_used=[],
            tool_results={},
            reasoning="",
            final_answer="",
            book_ids=None,
            error=None
        )
        
        try:
            # Execute the graph
            final_state = self.graph.invoke(initial_state)
            
            return AgentResult(
                target="respond",  # Override in subclasses
                text=final_state.get("final_answer", ""),
                success=not final_state.get("error"),
                tool_calls=self._format_tool_calls(final_state.get("tools_used", [])),
                book_ids=final_state.get("book_ids"),
                policy_version=self.policy_name,
            )
            
        except Exception as e:
            append_chatbot_log(f"[AGENT_ERROR] {str(e)}")
            return AgentResult(
                target="respond",
                text=f"I encountered an error: {str(e)}",
                success=False,
                tool_calls=[],
                policy_version=self.policy_name,
            )
    
    def _format_tool_calls(self, tools_used: List[dict]) -> List[ToolCall]:
        """Convert tools_used to ToolCall objects."""
        return [
            ToolCall(name=tool.get("name", ""), args=tool.get("args", {}))
            for tool in tools_used
        ]