# app/agents/web_agent.py
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

from app.agents.logging import get_logger, capture_agent_console_and_httpx
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

import re

def _extract_final_answer_from_error(e) -> str:
    """
    On a parsing error, DO NOT emit a Final Answer. Instruct the agent to continue.
    This keeps the chain alive so it can call return_book_ids next.
    """
    return (
        "PARSING_ERROR: You emitted prose or malformed output. "
        "Continue the chain. You MUST either call a tool next or, if you are done, "
        "first call return_book_ids with a JSON list of item_idx, then write Final Answer."
    )

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
        verbose=True,  # KEEP verbose=True so all the classic prints are produced
        handle_parsing_errors=_extract_final_answer_from_error,
        max_iterations=10,
        max_execution_time=300,
        early_stopping_method="force",
        return_intermediate_steps=True,
    )

def answer(question: str, current_user=None, db=None, user_num_ratings: Optional[int] = None) -> Dict[str, Any]:
    try:
        executor = _get_executor(current_user=current_user, db=db, user_num_ratings=user_num_ratings)

        # Capture EXACT verbose stream + httpx logs and append to logs/chatbot.log
        with capture_agent_console_and_httpx():
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
