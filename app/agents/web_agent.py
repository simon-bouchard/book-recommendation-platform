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

from app.agents.help_tools import help_tools
_tools = _tools + help_tools

# ---- Minimal ReAct agent with strict caps ----
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

_PROMPT = """You are a virtual librarian helping readers discover and understand books.

Voice & conduct:
- Warm, concise, librarian tone. Never invent titles or facts.
- If metadata is missing, say so rather than guessing.
- Never output literal placeholders like <your reply>—write real text.

Tool-use policy (very strict):
- Do NOT use tools, unless necessary. Answer directly if you can.
- For questions about **how this website works** (profile, ratings, recommendations, chatbot, limits, portfolio status, etc.):
  1) Use **help-search** with the user’s question to find the best doc section.
  2) If needed, follow up with **help-read** using the file suggested by help-search (e.g., 'books.md', 'profile.md', 'chatbot.md', 'overview.md', 'versioning.md', 'faq.md', 'recommendations.md').
- For external facts, links, BEST/TOP/LATEST/COMPARE lists, or fresh info, use web-search / Wikipedia / Open Library.
- NEVER use tools for greetings, chit-chat, or when the user says “don’t suggest”, “not yet”, “just talk”, “no recommendations”, etc.
- If the user message is short (≤ 12 words) and does not request site help or external facts, do not use tools.
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

Q: what is this website?
A: Thought: Site-purpose question; search docs.
Action: help-search
Action Input: what is this website
Observation: (top sections)
Action: help-read
Action Input: overview.md
Observation: (overview content)
A: Final Answer: This is a portfolio project that demonstrates a machine learning–powered book recommender. It is not a commercial product or production service.

Q: is this a real service/product?
A: Thought: Site-purpose question; search docs.
Action: help-search
Action Input: real service product portfolio
Observation: (top sections)
Action: help-read
Action Input: versioning.md
Observation: (versioning content)
A: Final Answer: No—this is a portfolio demonstration, not a commercial product. It showcases an end-to-end recommender with daily training.

Q: how does the book page work?
A: Thought: Feature question; search docs.
Action: help-search
Action Input: book page
Observation: (top sections including books.md)
Action: help-read
Action Input: books.md
Observation: (books content)
A: Final Answer: Each book page shows core metadata, lets you rate or mark as read with comments, and includes a Similar Books section with subject-based, ALS-based, and hybrid similarity options.

Q: how does the profile page work?
A: Thought: Feature question; search docs.
Action: help-search
Action Input: profile page
Observation: (top sections including profile.md)
Action: help-read
Action Input: profile.md
Observation: (profile content)
A: Final Answer: Your profile shows user info and stats, past interactions, and two recommendation sections—subject-based (with a subject/popularity slider) and ALS (unlocks after 10 ratings).

Q: Why don’t I get personalised recommendations if I rated 10 books?
A: Thought: Policy question; search docs.
Action: help-search
Action Input: 10 ratings personalised recommendations
Observation: (top sections including faq.md or recommendations.md)
Action: help-read
Action Input: faq.md
Observation: (FAQ content)
A: Final Answer: Even after 10 ratings, ALS-based personalised recs appear only after the next daily training run. Until then you’ll still see subject + popularity results.

Q: how does the chatbot work?
A: Thought: Chatbot question; search docs.
Action: help-search
Action Input: chatbot
Observation: (top sections including chatbot.md)
Action: help-read
Action Input: chatbot.md
Observation: (chatbot content)
A: Final Answer: The chatbot is a demo virtual librarian. In Web mode it can use external tools (DuckDuckGo, Wikipedia). Internal mode (planned) will connect to the ALS/Subject/LightGBM/Bayesian pipeline. Short conversation memory is kept via cookie + Redis (3 turns, ~2 days).
 
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
