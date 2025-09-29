# app/agents/infrastructure/response_agent.py
"""
Simple conversational agent with no tools - just LLM responses.
"""
from app.agents.domain.entities import (
    AgentConfiguration, 
    AgentCapability,
    AgentExecutionState,
    ExecutionStatus
)
from app.agents.prompts.loader import read_prompt
from .base_langgraph_agent import BaseLangGraphAgent
from langchain_core.messages import HumanMessage


class ResponseAgent(BaseLangGraphAgent):
    """
    Simple conversational agent with no tool access.
    Just provides direct LLM responses.
    """
    
    def __init__(self):
        # Build configuration
        configuration = AgentConfiguration(
            policy_name="persona.system.md",
            capabilities=frozenset([AgentCapability.CONVERSATIONAL]),
            allowed_tools=frozenset(),  # No tools
            llm_tier="small",
            timeout_seconds=20,
            max_iterations=1  # Single pass
        )
        
        super().__init__(configuration)
    
    def _get_system_prompt(self) -> str:
        """Load basic conversational prompt."""
        persona = read_prompt("persona.system.md")
        return persona.strip()
    
    def _get_target_category(self) -> str:
        return "respond"
    
    def _reason_node(self, state: AgentExecutionState) -> AgentExecutionState:
        """
        Override reason node for response agent - just generate answer directly.
        No tool decision making needed.
        """
        try:
            # Build simple prompt
            system_prompt = self._get_system_prompt()
            user_query = state.input_text
            
            # Add conversation history if available
            context_parts = [system_prompt]
            
            if state.conversation_history:
                context_parts.append("\nConversation History:")
                for turn in state.conversation_history[-3:]:
                    if 'u' in turn:
                        context_parts.append(f"User: {turn['u']}")
                    if 'a' in turn:
                        context_parts.append(f"Assistant: {turn['a']}")
            
            context_parts.append(f"\nUser: {user_query}")
            context_parts.append("\nProvide a helpful, conversational response.")
            
            full_prompt = "\n".join(context_parts)
            
            # Get LLM response
            messages = [HumanMessage(content=full_prompt)]
            response = self.llm.invoke(messages)
            
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # Store as final answer
            state.intermediate_outputs["final_answer"] = answer
            state.intermediate_outputs["next_action"] = {"type": "finalize"}
            state.add_reasoning_step(answer)
            
        except Exception as e:
            state.mark_failed(str(e))
            state.intermediate_outputs["final_answer"] = (
                "I'm having trouble responding right now."
            )
            state.intermediate_outputs["next_action"] = {"type": "finalize"}
        
        return state