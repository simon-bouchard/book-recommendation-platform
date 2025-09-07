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

def _get_executor():
    registry = ToolRegistry(
        web=True,
        help=True,
        gates=InternalToolGates(internal_enabled=False)
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
    )

def answer(question: str) -> str:
    try:
        executor = _get_executor()
        res = executor.invoke({"input": question}, return_intermediate_steps=True)
        out = ""
        if isinstance(res, dict):
            out = (res.get("output") or "").strip()
        elif isinstance(res, str):
            out = res.strip()

        if not out or out.lower().startswith("agent stopped"):
            return "Final Answer: I’m here and ready to chat—what would you like to talk about?"

        if not out.startswith("Final Answer:"):
            import re
            m = re.search(r"Final Answer:\s*(.+)", out, flags=re.S)
            if m:
                return "Final Answer: " + m.group(1).strip()
            else:
                return "Final Answer: " + out

        return out
    except Exception as e:
        logger.exception("Agent error: %s", e)
        return "Final Answer: I ran into an error finishing that—mind rephrasing?"
