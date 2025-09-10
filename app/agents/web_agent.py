from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

from app.agents.logging import get_logger
logger = get_logger(__name__)

# ---- LLM: OpenAI by default ----
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=30)
# Groq:
# from langchain_groq import ChatGroq
# _llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0, timeout=30)

from app.agents.prompts import AGENT_PROMPT
from app.agents.tools import ToolRegistry, InternalToolGates

def _get_executor(current_user=None, db=None, user_num_ratings: Optional[int] = None):
    registry = ToolRegistry(
        web=True,
        help=True,
        gates=InternalToolGates(internal_enabled=True,  # flip on when you want ML tools
                                user_num_ratings=user_num_ratings,
                                warm_threshold=10),
        ctx_user=current_user,
        ctx_db=db,
    )
    tools = registry.get_tools()
    agent = create_react_agent(_llm, tools, AGENT_PROMPT)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
        max_execution_time=300,
        early_stopping_method="force",
        return_intermediate_steps=True,
    )

def answer(question: str, current_user=None, db=None, user_num_ratings: Optional[int] = None) -> Dict[str, Any]:
    try:
        executor = _get_executor(current_user=current_user, db=db, user_num_ratings=user_num_ratings)
        res = executor.invoke({"input": question})

        # Visible text (keep your current behavior)
        out = ""
        if isinstance(res, dict):
            out = (res.get("output") or "").strip()
        elif isinstance(res, str):
            out = res.strip()

        if not out or out.lower().startswith("agent stopped"):
            out = "Final Answer: I’m here and ready to chat—what would you like to talk about?"

        if "Final Answer:" not in out:
            out = "Final Answer: " + out

        # Return text + raw intermediate steps (for runtime.py to process)
        steps: List = res.get("intermediate_steps") if isinstance(res, dict) else []
        return {"text": out, "intermediate_steps": steps}

    except Exception as e:
        logger.exception("Agent error: %s", e)
        return {"text": "Final Answer: I ran into an error finishing that—mind rephrasing?", "intermediate_steps": []}