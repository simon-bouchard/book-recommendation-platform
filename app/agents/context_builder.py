# app/agents/context_builder.py
from typing import List, Optional
from sqlalchemy.orm import Session
from app.agents.user_context import fetch_user_context, format_user_context

def build_composed_input(
    db: Optional[Session],
    user_id: Optional[int],
    use_profile: bool,
    history: List[dict],
    user_text: str,
    hist_turns: int = 3,
) -> str:
    sections: List[str] = []

    # Profile context
    if use_profile and user_id and db is not None:
        try:
            ctx = fetch_user_context(db, user_id, limit=15)
            block = format_user_context(ctx)
            sections.append(f"(User profile context — may guide your answer)\n{block}\n")
        except Exception:
            pass

    # Rolling history
    if history:
        ctx_lines: List[str] = []
        for turn in history[-hist_turns:]:
            u = turn.get("u")
            a = turn.get("a")
            if u:
                ctx_lines.append(f"User: {u}")
            if a:
                ctx_lines.append(f"Assistant: {a}")
        if ctx_lines:
            sections.append("(Conversation so far — use only for context)\n" + "\n".join(ctx_lines) + "\n")

    # Current user message
    sections.append(f"User: {user_text.strip()}")

    return "\n\n".join(sections)
