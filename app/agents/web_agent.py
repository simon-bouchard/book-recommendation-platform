# app/agents/web_agent.py
import os, re
from dotenv import load_dotenv

load_dotenv()

# ---- LLM: OpenAI by default; easy to swap to Groq later ----
# Set OPENAI_API_KEY in .env (or replace with Groq - see comment below).
from langchain_openai import ChatOpenAI
_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=30)

# If you prefer Groq now:
# from langchain_groq import ChatGroq
# _llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0, timeout=30)

# ---- Tools: DuckDuckGo web search + Wikipedia (simple & free) ----
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool

_ddg = DuckDuckGoSearchResults(num_results=5)
_wiki = WikipediaAPIWrapper(lang="en")

def _norm(q: str) -> str:
    toks = re.findall(r"\w+", q.lower())
    toks.sort()
    return " ".join(toks)

_seen = {"web-search": set(), "Wikipedia": set()}

def _safe_web(q: str) -> str:
    key = _norm(q)
    if key in _seen["web-search"]:
        return "[Guardrail] Repeated web search. Use what you have and answer."
    _seen["web-search"].add(key)
    out = _ddg.run(q)
    return out if out.strip() else "[Search] No results."

def _safe_wiki(q: str) -> str:
    key = _norm(q)
    if key in _seen["Wikipedia"]:
        return "[Guardrail] Repeated Wikipedia query. Use what you have and answer."
    _seen["Wikipedia"].add(key)
    out = _wiki.run(q)
    return out if out.strip() else "[Wiki] No content returned."

_tools = [
    Tool(
        name="web-search",
        func=_safe_web,
        description="General web search for fresh info, lists, comparisons, links."
    ),
    Tool(
        name="Wikipedia",
        func=_safe_wiki,
        description="Background/definitions only; avoid for 'best/top/latest' lists."
    ),
]

# ---- Minimal ReAct agent with strict caps ----
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

_PROMPT = """You are a virtual librarian helping readers discover and understand books.

Voice & conduct:
- Warm, concise, librarian tone. Never invent titles or facts.
- If metadata is missing, say so rather than guessing.
- Never output literal placeholders like <your reply>—write real text.

Tool-use policy (very strict):
- DEFAULT: Do NOT use tools. Answer directly if you can.
- Use a tool ONLY when the user explicitly asks for external facts, links, BEST/TOP/LATEST/COMPARE lists, or anything requiring up-to-date web info.
- NEVER use tools for greetings, chit-chat, or when the user says “don’t suggest”, “not yet”, “just talk”, “no recommendations”, etc.
- If the user message is short (≤ 12 words) and does not request external facts, do not use tools.
- If no tools are needed, skip Thought/Action/Observation and go straight to your final line.

You have access to the following tools:
{tools}

Action format (only when you DO use a tool):
Thought: <your reasoning>
Action: <one of [{tool_names}]>
Action Input: <the input string>
Observation: <the tool result>

Cap: Use at most 3 tools total. If none are needed, use zero.

Finalization rule:
Always end with exactly one line that begins with:
Final Answer:
Follow it immediately with your actual reply (no brackets or placeholders).

Few-shots:

Q: hi
A: Final Answer: Hi! I’m your virtual librarian—happy to chat. What kind of reading mood have you been in lately?

Q: hi, don't suggest me books yet—just talk
A: Final Answer: Of course—no recommendations yet. Do you tend to enjoy character-driven stories, fast plots, or something more contemplative?

Q: who are you?
A: Final Answer: I’m your virtual librarian. I can chat about books and, when you’re ready, search for options—no rush. What was the last book you enjoyed?

Question: {input}

{agent_scratchpad}
"""

_agent = create_react_agent(
    _llm,
    _tools,
    PromptTemplate.from_template(_PROMPT),
)

_executor = AgentExecutor(
    agent=_agent,
    tools=_tools,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=10,
    max_execution_time=300,
    early_stopping_method="force",
)

def answer(question: str) -> str:
    """
    Run the tiny web agent and return a short, readable answer.
    Always normalize to a single 'Final Answer:' line.
    """
    try:
        res = _executor.invoke({"input": question})
        out = ""
        if isinstance(res, dict):
            out = (res.get("output") or "").strip()
        elif isinstance(res, str):
            out = res.strip()

        # If the agent stopped without a final, or returned control text, synthesize a clean line.
        if not out or out.lower().startswith("agent stopped"):
            return "Final Answer: I’m here and ready to chat—what would you like to talk about?"

        # If the model printed a final somewhere in the blob, extract it.
        if not out.startswith("Final Answer:"):
            m = re.search(r"Final Answer:\s*(.+)", out, flags=re.S)
            if m:
                return "Final Answer: " + m.group(1).strip()
            else:
                return "Final Answer: " + out

        # Already in the right shape
        return out
    except Exception as e:
        return "Final Answer: I ran into an error finishing that—mind rephrasing?"
