# Add this to a new file: app/agents/llm_wrapper.py

from typing import Any, Dict, List, Optional, Union
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.prompt_values import PromptValue, StringPromptValue, ChatPromptValue


class SystemPromptLLM:
    """
    Wrapper that injects a system prompt into every LLM call.
    Works with both string inputs and message inputs.
    """
    
    def __init__(self, base_llm: BaseChatModel, system_prompt: str):
        self.base_llm = base_llm
        self.system_prompt = system_prompt.strip()
        
        # Copy essential attributes that agents expect
        self.model_name = getattr(base_llm, 'model_name', 'wrapped_model')
        self.model = getattr(base_llm, 'model', self.model_name)
        self.temperature = getattr(base_llm, 'temperature', 0.0)
        self.max_tokens = getattr(base_llm, 'max_tokens', None)
        
        # Copy any model_kwargs
        if hasattr(base_llm, 'model_kwargs'):
            self.model_kwargs = base_llm.model_kwargs
    
    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the base LLM."""
        if name in ['base_llm', 'system_prompt', 'model_name', 'model', 'temperature', 'max_tokens', 'model_kwargs']:
            return object.__getattribute__(self, name)
        return getattr(self.base_llm, name)
    
    def _prepare_messages(self, input_data: Any) -> List[BaseMessage]:
        """Convert input to messages and prepend system message."""
        messages = []
        
        # Handle different input types
        if isinstance(input_data, str):
            # Plain string input
            messages = [HumanMessage(content=input_data)]
        elif isinstance(input_data, StringPromptValue):
            # String prompt value from LangChain
            messages = [HumanMessage(content=input_data.text)]
        elif isinstance(input_data, ChatPromptValue):
            # Chat prompt value - already has messages
            messages = input_data.messages
        elif isinstance(input_data, list):
            # List of messages
            messages = input_data
        elif hasattr(input_data, 'to_messages'):
            # PromptValue that can be converted
            messages = input_data.to_messages()
        else:
            # Fallback - treat as string
            messages = [HumanMessage(content=str(input_data))]
        
        # Check if system message already exists
        has_system = any(isinstance(msg, SystemMessage) for msg in messages)
        
        if not has_system and self.system_prompt:
            # Prepend system message
            messages = [SystemMessage(content=self.system_prompt)] + messages
        
        return messages
    
    def invoke(self, 
               input: Union[PromptValue, str, List[BaseMessage]], 
               config: Optional[Dict] = None, 
               **kwargs: Any) -> BaseMessage:
        """Main invoke method that handles system prompt injection."""
        messages = self._prepare_messages(input)
        return self.base_llm.invoke(messages, config=config, **kwargs)
    
    def generate(self,
                 messages: List[List[BaseMessage]],
                 stop: Optional[List[str]] = None,
                 callbacks: Optional[List] = None,
                 **kwargs: Any) -> ChatResult:
        """Handle batch generation."""
        processed_messages = []
        for message_list in messages:
            processed_messages.append(self._prepare_messages(message_list))
        
        return self.base_llm.generate(
            processed_messages, 
            stop=stop, 
            callbacks=callbacks, 
            **kwargs
        )
    
    async def ainvoke(self,
                      input: Union[PromptValue, str, List[BaseMessage]],
                      config: Optional[Dict] = None,
                      **kwargs: Any) -> BaseMessage:
        """Async version of invoke."""
        messages = self._prepare_messages(input)
        return await self.base_llm.ainvoke(messages, config=config, **kwargs)


def create_system_prompt_llm(base_llm: BaseChatModel, system_prompt: str) -> SystemPromptLLM:
    """Factory function to create a system prompt wrapper."""
    return SystemPromptLLM(base_llm, system_prompt)