from datetime import datetime
from typing import List, Optional, Union

from dotenv import load_dotenv
from fastapi import (
    APIRouter,
    Body,
    Depends,
    HTTPException,
    Query,
    Request,
    status,
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import case, desc, func
from sqlalchemy.orm import Session

from app.agents.logging import get_logger
from app.database import get_db
from app.models import clean_float_values, get_all_subject_counts
from app.search.engine import SearchEngine
from app.search.models import SearchMode, SearchRequest
from app.search.search_utils import _build_search_request, update_book_ratings_in_meili
from app.table_models import Book, Interaction, Subject, User, UserFavSubject
from metrics import RATING_ACTIONS
from models.client.registry import get_als_client, get_similarity_client
from routes.auth import get_current_user

logger = get_logger(__name__)

search_engine = SearchEngine()

router = APIRouter()
templates = Jinja2Templates(directory="templates")
templates.env.globals["now"] = datetime.utcnow

load_dotenv()


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    error = request.session.pop("flash_error", None)
    warning = request.session.pop("flash_warning", None)
    next_url = request.query_params.get("next", "")
    return templates.TemplateResponse(
        "login_shell.html",
        {"request": request, "next": next_url, "error": error or "", "warning": warning or ""},
    )


@router.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request):
    return templates.TemplateResponse("signup_shell.html", {"request": request, "error": ""})


@router.get("/profile")
def profile_page(
    request: Request, current_user: dict = Depends(get_current_user), db: Session = Depends(get_db)
):
    if not current_user:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

    # Efficient SQL aggregation: count total and rated
    counts = (
        db.query(
            func.count().label("total"),
            func.count(case((Interaction.rating.isnot(None), 1))).label("rated"),
        )
        .filter(Interaction.user_id == current_user.user_id)
        .one()
    )

    num_books_read = counts.total
    num_ratings = counts.rated

    {
        "id": current_user.user_id,
        "username": current_user.username,
        "email": current_user.email,
        "num_books_read": num_books_read,
        "num_ratings": num_ratings,
        "favorite_subjects": [
            s.subject.subject
            for s in current_user.favorite_subjects
            if s.subject and s.subject.subject != "[NO_SUBJECT]"
        ],
    }

    return templates.TemplateResponse("profile_shell.html", {"request": request})


@router.get("/profile/json")
def profile_json(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    counts = (
        db.query(
            func.count().label("total"),
            func.count(case((Interaction.rating.isnot(None), 1))).label("rated"),
        )
        .filter(Interaction.user_id == current_user.user_id)
        .one()
    )

    return {
        "id": current_user.user_id,
        "username": current_user.username,
        "email": current_user.email,
        "num_books_read": counts.total,
        "num_ratings": counts.rated,
        "num_interactions": counts.total,
        "favorite_subjects": [
            s.subject.subject
            for s in current_user.favorite_subjects
            if s.subject and s.subject.subject != "[NO_SUBJECT]"
        ],
    }


@router.get("/profile/is_warm")
async def profile_is_warm(current_user=Depends(get_current_user)):
    """
    Check whether the current user has ALS factors (warm/cold gate).

    Separated from the profile page so the sync page route has no async
    dependency. The frontend fetches this after page load and updates the
    UI accordingly. Returns is_warm=False on any model server error so the
    UI degrades gracefully if the ALS server is unavailable.
    """
    if not current_user:
        return {"is_warm": False}
    try:
        resp = await get_als_client().has_als_user(current_user.user_id)
        return {"is_warm": resp.is_warm}
    except Exception:
        return {"is_warm": False}


@router.post("/profile/update")
async def update_profile(
    current_user: User = Depends(get_current_user),
    data: dict = Body(...),
    db: Session = Depends(get_db),
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    user = db.query(User).filter(User.user_id == current_user.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.username = data.get("username", user.username)
    user.email = data.get("email", user.email)

    # Update favorite subjects
    new_subjects = data.get("favorite_subjects", [])
    if isinstance(new_subjects, list):
        db.query(UserFavSubject).filter(UserFavSubject.user_id == user.user_id).delete()
        for subject in new_subjects:
            subj_obj = db.query(Subject).filter(Subject.subject == subject).first()
            if subj_obj:
                db.add(UserFavSubject(user_id=user.user_id, subject_idx=subj_obj.subject_idx))

    db.commit()
    return {"message": "Profile updated"}


@router.delete("/profile")
def delete_profile(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Re-fetch as ORM row to be safe
    user = db.query(User).filter(User.user_id == current_user.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Remove dependents explicitly (don’t rely on DB cascade)
    db.query(UserFavSubject).filter(UserFavSubject.user_id == user.user_id).delete(
        synchronize_session=False
    )
    db.query(Interaction).filter(Interaction.user_id == user.user_id).delete(
        synchronize_session=False
    )

    # Delete user
    db.delete(user)
    db.commit()

    return {"message": "Profile deleted"}


@router.get("/profile/ratings")
def get_user_ratings(
    limit: int = Query(10, ge=1, le=100),
    cursor: Optional[int] = Query(None, description="Keyset: last seen Interaction.id"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # total count (for header)
    total_count = (
        db.query(func.count(Interaction.id))
        .filter(Interaction.user_id == current_user.user_id)
        .scalar()
        or 0
    )

    q = (
        db.query(Interaction, Book)
        .join(Book, Book.item_idx == Interaction.item_idx)
        .filter(Interaction.user_id == current_user.user_id)
    )
    if cursor is not None:
        # simple, stable keyset: fetch strictly older ids
        q = q.filter(Interaction.id < cursor)

    rows = (
        q.order_by(desc(Interaction.id))  # newest first, stable by id
        .limit(limit + 1)  # +1 to detect has_more
        .all()
    )

    items = []
    for inter, book in rows[:limit]:
        author_obj = getattr(book, "author", None)
        author_str = getattr(author_obj, "name", None)
        if not isinstance(author_str, str) or not author_str.strip():
            author_str = "Unknown Author"

        cover_url_small = (
            f"https://covers.openlibrary.org/b/id/{getattr(book, 'cover_id', None)}-S.jpg"
            if getattr(book, "cover_id", None)
            else "/static/img/placeholder_cover.png"
        )
        item = {
            "book_id": book.item_idx,
            "title": book.title or "Untitled",
            "author": author_str,
            "year": getattr(book, "year", None),
            "cover_url_small": cover_url_small,
            "rated_at": inter.timestamp.isoformat() if inter.timestamp else None,
        }
        if getattr(inter, "rating", None) is not None:
            item["user_rating"] = inter.rating
        if getattr(inter, "comment", None) is not None:
            item["comment"] = inter.comment or ""
        items.append(item)

    has_more = len(rows) > limit
    next_cursor = None
    if has_more:
        last_inter, _ = rows[limit - 1]
        next_cursor = int(last_inter.id)  # next page continues from this id

    return {
        "items": items,
        "total_count": total_count,
        "has_more": has_more,
        "next_cursor": next_cursor,
    }


@router.post("/rating")
async def new_rating(
    current_user=Depends(get_current_user), data: dict = Body(...), db: Session = Depends(get_db)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    item_idx = data.get("item_idx")
    rating = data.get("rating")  # Can be None for 'read'
    comment = data.get("comment")

    book = db.query(Book).filter(Book.item_idx == item_idx).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    interaction_type = "rated" if rating is not None else "read"
    now = datetime.utcnow()

    interaction = (
        db.query(Interaction)
        .filter(Interaction.user_id == current_user.user_id, Interaction.item_idx == book.item_idx)
        .first()
    )

    if interaction:
        interaction.rating = rating
        interaction.comment = comment
        interaction.type = interaction_type
        interaction.timestamp = now
    else:
        interaction = Interaction(
            user_id=current_user.user_id,
            item_idx=book.item_idx,
            rating=rating,
            comment=comment,
            type=interaction_type,
            timestamp=now,
        )
        db.add(interaction)

    db.commit()
    db.close()

    RATING_ACTIONS.labels(action=interaction_type).inc()

    update_book_ratings_in_meili(item_idx)

    return {"message": "Interaction recorded successfully"}


@router.delete("/rating/{item_idx}")
async def delete_rating(
    item_idx: int, current_user=Depends(get_current_user), db: Session = Depends(get_db)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    interaction = (
        db.query(Interaction)
        .filter(Interaction.user_id == current_user.user_id, Interaction.item_idx == item_idx)
        .first()
    )

    if not interaction:
        # Idempotent delete; OK if there's nothing to remove
        return {"message": "No rating found to delete"}

    db.delete(interaction)
    db.commit()

    RATING_ACTIONS.labels(action="delete").inc()

    update_book_ratings_in_meili(item_idx)

    return {"message": "Rating deleted"}


@router.get("/book/{item_idx}", response_class=HTMLResponse)
def book_recommendation(
    request: Request,
    item_idx: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    book = db.query(Book).filter(Book.item_idx == item_idx).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    # Average rating
    average = (
        db.query(func.avg(Interaction.rating))
        .filter(Interaction.item_idx == book.item_idx, Interaction.rating.isnot(None))
        .scalar()
    )

    # Count of ratings
    rating_count = (
        db.query(Interaction)
        .filter(Interaction.item_idx == book.item_idx, Interaction.rating.isnot(None))
        .count()
    )

    # Subjects list
    subject_ids = [s.subject_idx for s in book.subjects]
    subject_names = db.query(Subject.subject).filter(Subject.subject_idx.in_(subject_ids)).all()
    subjects = [s.subject for s in subject_names]

    any(s != "[NO_SUBJECT]" for s in subjects)

    {
        "isbn": book.isbn,
        "item_idx": book.item_idx,
        "title": book.title,
        "author": book.author.name if book.author else "Unknown",
        "year": book.year,
        "description": book.description,
        "cover_id": book.cover_id,
        "average_rating": round(average, 2) if average else None,
        "rating_count": rating_count,
        "subjects": subjects,
        "num_pages": book.num_pages,
        "page": "book",
    }

    if current_user:
        interaction = (
            db.query(Interaction)
            .filter(
                Interaction.user_id == current_user.user_id, Interaction.item_idx == book.item_idx
            )
            .first()
        )
        if interaction and (interaction.rating is not None or interaction.comment):
            pass

    return templates.TemplateResponse(
        "book_shell.html",
        {"request": request, "item_idx": item_idx},
    )


@router.get("/book/{item_idx}/json")
def book_json(
    item_idx: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    book = db.query(Book).filter(Book.item_idx == item_idx).first()
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    average = (
        db.query(func.avg(Interaction.rating))
        .filter(Interaction.item_idx == book.item_idx, Interaction.rating.isnot(None))
        .scalar()
    )
    rating_count = (
        db.query(Interaction)
        .filter(Interaction.item_idx == book.item_idx, Interaction.rating.isnot(None))
        .count()
    )

    subject_ids = [s.subject_idx for s in book.subjects]
    subject_names = db.query(Subject.subject).filter(Subject.subject_idx.in_(subject_ids)).all()
    subjects = [s.subject for s in subject_names]
    has_real_subjects = any(s != "[NO_SUBJECT]" for s in subjects)
    filtered_subjects = [s for s in subjects if s != "[NO_SUBJECT]"]

    user_rating = None
    if current_user:
        interaction = (
            db.query(Interaction)
            .filter(
                Interaction.user_id == current_user.user_id, Interaction.item_idx == book.item_idx
            )
            .first()
        )
        if interaction and (interaction.rating is not None or interaction.comment):
            user_rating = {"rating": interaction.rating, "comment": interaction.comment}

    return {
        "book": {
            "item_idx": book.item_idx,
            "title": book.title,
            "author": book.author.name if book.author else "Unknown",
            "year": book.year,
            "description": book.description,
            "cover_id": book.cover_id,
            "isbn": book.isbn,
            "average_rating": round(average, 2) if average else None,
            "rating_count": rating_count,
            "subjects": filtered_subjects,
            "num_pages": book.num_pages,
        },
        "user_rating": user_rating,
        "has_real_subjects": has_real_subjects,
        "logged_in": current_user is not None,
    }


@router.get("/book/{item_idx}/has_als")
async def book_has_als(item_idx: int):
    """
    Check whether a book has ALS factors available.

    Separated from the book page so the sync page route has no async
    dependency. The frontend fetches this after page load and shows or
    hides behavioral similarity controls accordingly. Returns has_als=False
    on any model server error so the UI degrades gracefully.
    """
    try:
        resp = await get_similarity_client().has_book_als(item_idx)
        return {"has_als": resp.has_als}
    except Exception:
        return {"has_als": False}


@router.get("/book/{item_idx}/comments")
def get_book_comments_paginated(
    item_idx: int,
    limit: int = Query(10, ge=1, le=100),
    cursor: Optional[int] = Query(None),
    db: Session = Depends(get_db),
):
    total_count = (
        db.query(func.count(Interaction.id))
        .filter(
            Interaction.item_idx == item_idx,
            Interaction.comment.isnot(None),
            Interaction.comment != "",
        )
        .scalar()
        or 0
    )

    q = (
        db.query(Interaction, User)
        .join(User, User.user_id == Interaction.user_id)
        .filter(
            Interaction.item_idx == item_idx,
            Interaction.comment.isnot(None),
            Interaction.comment != "",
        )
    )
    if cursor is not None:
        q = q.filter(Interaction.id < cursor)

    rows = q.order_by(desc(Interaction.id)).limit(limit + 1).all()

    items = []
    for inter, user in rows[:limit]:
        items.append(
            {
                "user_id": user.user_id,
                "username": user.username,
                "rating": inter.rating,
                "comment": inter.comment or "",
                "rated_at": inter.timestamp.isoformat() if inter.timestamp else None,
            }
        )

    has_more = len(rows) > limit
    if has_more:
        last_inter, _ = rows[limit - 1]
        int(last_inter.id)

    return {
        "items": items,
        "total_count": total_count,
        "has_more": has_more,
    }


@router.get("/search", response_class=HTMLResponse)
def search_books(request: Request):
    return templates.TemplateResponse("search_shell.html", {"request": request, "page": "search"})


@router.get("/search/json")
def search_books_json(
    query: str = Query(""),
    subjects: Optional[str] = Query(None),
    page: int = Query(0, ge=0),
    page_size: int = Query(60, ge=1, le=200),
    mode: SearchMode = Query(SearchMode.MEILI, description="Currently only 'meili' is supported"),
    sort: Optional[str] = Query(
        "bayes_pop:desc", regex=r"^[\w]+:(asc|desc)$", description="e.g. bayes_pop:desc"
    ),
    highlight: bool = Query(False),
    crop: Union[bool, int] = Query(False),
    min_score: Optional[float] = Query(None, ge=0, le=1),
    db: Session = Depends(get_db),
):
    """
    Fully-featured search API exposing all MeiliSearch capabilities.
    """
    search_req = _build_search_request(
        query=query,
        subjects=subjects,
        page=page,
        page_size=page_size + 1,  # +1 to detect has_more accurately
        mode=mode,
        sort=sort,
        highlight=highlight,
        crop=crop,
        min_score=min_score,
        db=db,
    )

    search_response = search_engine.search(search_req)

    results = search_response.results
    has_more = len(results) > page_size
    results = results[:page_size]

    total_pages = (search_response.total + page_size - 1) // page_size

    return {
        "results": clean_float_values([r.dict() for r in results]),
        "pagination": {
            "current_page": page,
            "page_size": page_size,
            "total_results": search_response.total,
            "total_pages": total_pages,
            "has_next": has_more or (page + 1) * page_size < search_response.total,
            "has_prev": page > 0,
        },
        "applied": {
            "mode": mode.value,
            "sort": sort,
            "highlight": highlight,
            "crop": crop if isinstance(crop, int) else None,
            "min_score": min_score,
        },
    }


@router.get("/search/autocomplete")
def autocomplete_suggestions(
    q: str = Query(..., min_length=2, description="Search query for autocomplete"),
    limit: int = Query(5, ge=1, le=10),
    highlight: bool = Query(False, description="Enable highlight formatting"),
    attributes_to_retrieve: Optional[List[str]] = Query(
        None, description="Fields to return. Default: ['item_idx', 'title', 'author']"
    ),
    db: Session = Depends(get_db),
):
    """
    Fast typo-tolerant autocomplete using Meilisearch
    Returns book titles and authors as suggestions
    """
    # Set default attributes if none provided
    if attributes_to_retrieve is None:
        attributes_to_retrieve = ["item_idx", "title", "author"]

    search_req = SearchRequest(
        query=q,
        mode=SearchMode.MEILI,
        page=0,
        page_size=limit,
        highlight=highlight,
        crop=False,  # No cropping for autocomplete
        attributes_to_retrieve=attributes_to_retrieve,
    )

    search_response = search_engine.search(search_req)

    suggestions = []
    for result in search_response.results:
        suggestion_data = {
            "item_idx": result.item_idx,
            "title": result.title,
            "author": result.author,
            "type": "book",
        }

        # Include highlighted fields if available and requested
        if highlight and hasattr(result, "description_snippet") and result.description_snippet:
            suggestion_data["highlighted_description"] = result.description_snippet

        suggestions.append(suggestion_data)

    return {"suggestions": suggestions}


@router.get("/subjects/suggestions")
def subject_suggestions(q: Optional[str] = Query(default=None), db: Session = Depends(get_db)):
    """
    Return subject suggestions:
    - If q is None → return top 20 subjects
    - If q is provided → return subjects that contain q (case-insensitive), sorted by count
    """
    all_subjects = get_all_subject_counts(db)

    if q:
        q_lower = q.lower()
        filtered = [s for s in all_subjects if q_lower in s["subject"].lower()]
        return {"subjects": filtered[:20]}
    else:
        return {"subjects": all_subjects[:20]}


@router.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)

    response.delete_cookie(key="access_token")

    return response
