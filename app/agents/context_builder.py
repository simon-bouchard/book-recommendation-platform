# app/agents/context_builder.py
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from app.agents.user_context import fetch_user_context, format_user_context
from app.agents.schemas import TurnInput

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

def get_router_view(
    history: List[Dict],
    k_user: int = 2,
    max_chars: int = 2000,
) -> str:
    """
    Return a compact slice for routing: the last K USER utterances only (newest last),
    joined by newlines. No assistant messages, no profile, hard-capped by max_chars.
    """
    if not history or k_user <= 0:
        return ""

    # Collect last k_user non-empty user messages, newest→oldest
    collected: List[str] = []
    for turn in reversed(history):
        u = (turn.get("u") or "").strip()
        if u:
            collected.append(u)
            if len(collected) >= k_user:
                break

    # Reverse to oldest→newest; join
    lines = list(reversed(collected))
    text = "\n".join(lines)

    # Enforce hard cap (trim from the front to keep most recent context intact)
    if max_chars > 0 and len(text) > max_chars:
        text = text[-max_chars:]

    return text


def get_branch_view(
    history: List[Dict],
    hist_turns: int,
) -> List[Dict]:
    """
    Return the last N (user, assistant) turns for branch composition.
    Structure matches what build_composed_input expects.
    """
    if not history or hist_turns <= 0:
        return []
    return history[-hist_turns:]

# context_builder.py
def make_router_input(history: list[dict], user_text: str, k_user: int = 2) -> TurnInput:
    short_view = get_router_view(history, k_user=k_user).strip()
    current = user_text.strip()

    if short_view:
        merged = (
            "[LAST_USER_MESSAGES]\n"
            f"{short_view}\n"
            "[/LAST_USER_MESSAGES]\n"
            "[CURRENT]\n"
            f"{current}\n"
            "[/CURRENT]"
        )
    else:
        merged = current

    return TurnInput(
        user_text=merged,
        full_history=[],
    )

def make_branch_input(
    history: list[dict],
    user_text: str,
    hist_turns: int,
    use_profile: bool,
    user_num_ratings: int | None,
    db=None,
    current_user=None,
    conv_id=None,
    uid=None,
) -> TurnInput:
    """
    Standard TurnInput for branch agents.
    """
    full_view = get_branch_view(history, hist_turns=hist_turns)
    return TurnInput(
        user_text=user_text.strip(),
        full_history=full_view,
        profile_allowed=use_profile,
        user_num_ratings=user_num_ratings,
        ctx={
            "db": db,
            "current_user": current_user,
            "conv_id": conv_id,
            "uid": uid,
        },
    )