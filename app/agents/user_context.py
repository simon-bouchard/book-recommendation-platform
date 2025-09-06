from typing import Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import desc
from app.table_models import User, UserFavSubject, Subject, Interaction, Book

def fetch_user_context(db: Session, user_id: int, limit: int = 15) -> Dict[str, Any]:
    """Return favorite subjects and recent interactions for a user."""
    ctx: Dict[str, Any] = {"fav_subjects": [], "interactions": []}

    # --- Favorite subjects ---
    favs = (
        db.query(Subject.subject)
          .join(UserFavSubject, Subject.subject_idx == UserFavSubject.subject_idx)
          .filter(UserFavSubject.user_id == user_id)
          .all()
    )
    ctx["fav_subjects"] = [s[0] for s in favs if s[0] != "[NO_SUBJECT]"]

    # --- Recent interactions with book info ---
    rows = (
        db.query(Interaction, Book)
          .join(Book, Book.item_idx == Interaction.item_idx)
          .filter(Interaction.user_id == user_id)
          .order_by(desc(Interaction.timestamp))
          .limit(limit)
          .all()
    )
    inters: List[Dict[str, Any]] = []
    for inter, book in rows:
        inters.append({
            "title": (book.title or "Untitled")[:70],
            "rating": inter.rating,
            "comment": (inter.comment or "")[:140],
            "date": inter.timestamp.isoformat() if inter.timestamp else None,
        })
    ctx["interactions"] = inters
    return ctx

def format_user_context(ctx: Dict[str, Any]) -> str:
    """Turn the dict into a text block the LLM can use."""
    favs = ", ".join(ctx.get("fav_subjects", [])[:3]) or "None"
    lines = [f"Favorite subjects: {favs}"]
    if ctx["interactions"]:
        lines.append("Recent interactions:")
        for x in ctx["interactions"][:10]:
            lines.append(
                f"- {x['title']} | rating={x['rating']} | {x['date']} | {x['comment']}"
            )
    else:
        lines.append("Recent interactions: None")
    return "\n".join(lines)
