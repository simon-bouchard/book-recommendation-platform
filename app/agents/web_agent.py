import os, re
from dotenv import load_dotenv

load_dotenv()

# ---- LLM: OpenAI by default ----
from langchain_openai import ChatOpenAI
_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=30)

# Groq:
# from langchain_groq import ChatGroq
# _llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0, timeout=30)

# ---- Tools: DuckDuckGo web search + Wikipedia (simple & free) ----
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool
import requests

_ddg = DuckDuckGoSearchResults(num_results=5)
_wiki = WikipediaAPIWrapper(lang="en")
_ol_session = requests.Session()

def _norm(q: str) -> str:
    toks = re.findall(r"\w+", q.lower())
    toks.sort()
    return " ".join(toks)

_seen = {"web-search": set(), "Wikipedia": set()}
_seen["openlibrary"] = set()
_seen["ol-work"] = set()

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

def _safe_ol_search(q: str) -> str:
    """Search Open Library for books; return a compact, readable list."""
    key = _norm(q)
    if key in _seen["openlibrary"]:
        return "[Guardrail] Repeated Open Library search. Use prior results."
    _seen["openlibrary"].add(key)

    try:
        # Simple, deterministic search
        r = _ol_session.get(
            "https://openlibrary.org/search.json",
            params={"q": q, "fields": "title,author_name,first_publish_year,key,edition_count", "limit": 5},
            timeout=12
        )
        r.raise_for_status()
        js = r.json()
        docs = js.get("docs", [])[:5]
        if not docs:
            return "[OpenLibrary] No results."
        lines = []
        for d in docs:
            title = d.get("title") or "Untitled"
            author = ", ".join(d.get("author_name", [])[:2]) or "Unknown"
            year = d.get("first_publish_year")
            work = d.get("key")  # like '/works/OL12345W'
            editions = d.get("edition_count")
            meta = []
            if year: meta.append(str(year))
            if editions: meta.append(f"{editions} eds")
            meta_str = f" ({', '.join(meta)})" if meta else ""
            lines.append(f"- {title} — {author}{meta_str}  [source: openlibrary.org{work}]")
        return "\n".join(lines)
    except Exception:
        return "[OpenLibrary] Error reaching API."

def _safe_ol_work(work_key: str) -> str:
    """Fetch a single work's metadata/description by key like '/works/OL12345W'."""
    key = _norm(work_key)
    if key in _seen["ol-work"]:
        return "[Guardrail] Repeated Open Library work fetch. Use prior results."
    _seen["ol-work"].add(key)

    # normalize input to '/works/..' form
    wk = work_key.strip()
    if not wk.startswith("/works/"):
        wk = f"/works/{wk}"

    try:
        r = _ol_session.get(f"https://openlibrary.org{wk}.json", timeout=12)
        r.raise_for_status()
        w = r.json()
        title = w.get("title") or "Untitled"
        desc = w.get("description")
        if isinstance(desc, dict):
            desc = desc.get("value")
        if not desc:
            desc = "[No description available]"
        return f"Title: {title}\nWork: {wk}\nDescription: {desc[:800]}{'…' if len(desc or '')>800 else ''}\n[source: openlibrary.org{wk}]"
    except Exception:
        return "[OpenLibrary] Error fetching work details."

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
    Tool(
        name="openlibrary-search",
        func=_safe_ol_search,
        description="Search Open Library for real book results (title/author/year). Input: plain query."
    ),
    Tool(
        name="openlibrary-work",
        func=_safe_ol_work,
        description="Fetch details for a specific Open Library work. Input: '/works/OLXXXXW' or just 'OLXXXXW'."
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
