import pandas as pd
from fastapi import APIRouter, HTTPException, Form, Request, FastAPI, Depends, status, Body, Query, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import os
from dotenv import load_dotenv
from datetime import datetime

from datetime import datetime
from sqlalchemy import func, desc, case
from sqlalchemy.orm import Session, joinedload
import traceback
import pycountry
from typing import List, Optional, Union

from routes.auth import get_current_user
from app.database import get_db
from app.table_models import Book, User, Interaction, BookSubject, Subject, UserFavSubject
from app.models import get_all_subject_counts, clean_float_values
from app.search.search_utils import get_search_results, update_book_ratings_in_meili
from models.book_similarity_engine import get_similarity_strategy
from models.recommender_strategy import RecommenderStrategy, WarmRecommender, ColdRecommender
from models.shared_utils import PAD_IDX, ModelStore
from app.search.models import SearchRequest, SearchMode
from app.search.engine import SearchEngine
from app.search.search_utils import _build_search_request

from app.agents.logging import get_logger
logger = get_logger(__name__)

search_engine = SearchEngine()

router = APIRouter()
templates = Jinja2Templates(directory="templates")
templates.env.globals['now'] = datetime.utcnow

load_dotenv()

@router.get('/login', response_class=HTMLResponse)
def signup_page(request: Request):
    countries = sorted([c.name for c in pycountry.countries])
    message = request.session.pop("flash_success", None)
    error = request.session.pop("flash_error", None)
    warning = request.session.pop("flash_warning", None)

    return templates.TemplateResponse("login.html", {
        "request": request,
        "countries": countries,
        "page": "login",
        "message": message,
        "error": error,
        "warning": warning
    })

@router.get('/profile')
def profile_page(request: Request, current_user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user:
        return RedirectResponse(url='/login', status_code=status.HTTP_303_SEE_OTHER)

    # Efficient SQL aggregation: count total and rated
    counts = db.query(
        func.count().label("total"),
        func.count(case((Interaction.rating.isnot(None), 1))).label("rated")
    ).filter(
        Interaction.user_id == current_user.user_id
    ).one()

    num_books_read = counts.total
    num_ratings = counts.rated

    # Get warm status from user_meta
    user_meta = ModelStore().get_user_meta()
    row = user_meta.loc[current_user.user_id] if current_user.user_id in user_meta.index else None
    ratings_from_pkl = int(row["user_num_ratings"]) if row is not None else 0
    is_warm = ratings_from_pkl >= 10

    user_data = {
        "id": current_user.user_id,
        "username": current_user.username,
        "email": current_user.email,
        "age": current_user.age,
        "country": current_user.country,
        "num_books_read": num_books_read,
        "num_ratings": num_ratings,
        "is_warm": is_warm,
        "favorite_subjects": [
            s.subject.subject for s in current_user.favorite_subjects
            if s.subject and s.subject.subject != "[NO_SUBJECT]"
        ],
    }

    message = request.session.pop("flash_success", None)
    return templates.TemplateResponse('profile.html', {
        'request': request,
        'user': user_data,
        "page": "profile",
        'message': message
    })

@router.post("/profile/update")
async def update_profile(
    current_user: User = Depends(get_current_user),
    data: dict = Body(...),
    db: Session = Depends(get_db)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    user = db.query(User).filter(User.user_id == current_user.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.username = data.get("username", user.username)
    user.email = data.get("email", user.email)
    user.age = data.get("age", user.age)
    user.country = data.get("country", user.country)

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
def delete_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Re-fetch as ORM row to be safe
    user = db.query(User).filter(User.user_id == current_user.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Remove dependents explicitly (don’t rely on DB cascade)
    db.query(UserFavSubject).filter(UserFavSubject.user_id == user.user_id).delete(synchronize_session=False)
    db.query(Interaction).filter(Interaction.user_id == user.user_id).delete(synchronize_session=False)

    # Delete user
    db.delete(user)
    db.commit()

    return {"message": "Profile deleted"}

@router.get("/profile/ratings")
def get_user_ratings(
        limit: int = Query(10, ge=1, le=100),
        cursor: Optional[int] = Query(None, description="Keyset: last seen Interaction.id"),
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
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
            q.order_by(desc(Interaction.id))       # newest first, stable by id
             .limit(limit + 1)                     # +1 to detect has_more
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
                if getattr(book, "cover_id", None) else "/static/img/placeholder_cover.png"
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
            next_cursor = int(last_inter.id)       # next page continues from this id

        return {
            "items": items,
            "total_count": total_count,
            "has_more": has_more,
            "next_cursor": next_cursor
        }

@router.post('/rating')
async def new_rating(current_user = Depends(get_current_user), data: dict = Body(...), db: Session = Depends(get_db)):

    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    item_idx = data.get('item_idx')
    rating = data.get('rating')  # Can be None for 'read'
    comment = data.get('comment')

    book = db.query(Book).filter(Book.item_idx == item_idx).first()
    if not book:
        raise HTTPException(status_code=404, detail='Book not found')

    interaction_type = 'rated' if rating is not None else 'read'
    now = datetime.utcnow()

    interaction = db.query(Interaction).filter(
        Interaction.user_id == current_user.user_id,
        Interaction.item_idx == book.item_idx
    ).first()

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
            timestamp=now
        )
        db.add(interaction)

    db.commit()
    db.close()

    update_book_ratings_in_meili(item_idx)

    return {'message': 'Interaction recorded successfully'}

@router.delete("/rating/{item_idx}")
async def delete_rating(
    item_idx: int,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    interaction = db.query(Interaction).filter(
        Interaction.user_id == current_user.user_id,
        Interaction.item_idx == item_idx
    ).first()

    if not interaction:
        # Idempotent delete; OK if there's nothing to remove
        return {"message": "No rating found to delete"}

    db.delete(interaction)
    db.commit()

    update_book_ratings_in_meili(item_idx)

    return {"message": "Rating deleted"}


@router.get('/book/{item_idx}', response_class=HTMLResponse)
async def book_recommendation(request: Request, item_idx: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):

    book = db.query(Book).filter(Book.item_idx == item_idx).first()   
    if not book:
        raise HTTPException(status_code=404, detail='Book not found')

    # Average rating
    average = db.query(func.avg(Interaction.rating)).filter(
        Interaction.item_idx == book.item_idx,
        Interaction.rating.isnot(None)
    ).scalar()

    # Count of ratings
    rating_count = db.query(Interaction).filter(
        Interaction.item_idx == book.item_idx,
        Interaction.rating.isnot(None)
    ).count()

    # Subjects list
    subject_ids = [s.subject_idx for s in book.subjects]
    subject_names = db.query(Subject.subject).filter(Subject.subject_idx.in_(subject_ids)).all()
    subjects = [s.subject for s in subject_names]

    has_real_subjects = any(s != "[NO_SUBJECT]" for s in subjects)

    book_info = {
        'isbn': book.isbn,
        'item_idx': book.item_idx,
        'title': book.title,
        'author': book.author.name if book.author else 'Unknown',
        'year': book.year,
        'description': book.description,
        'cover_id': book.cover_id,
        'average_rating': round(average, 2) if average else None,
        'rating_count': rating_count,
        'subjects': subjects,
        'num_pages': book.num_pages,
        'page': 'book',
    }

    user_rating = None
    if current_user:
        interaction = db.query(Interaction).filter(
            Interaction.user_id == current_user.user_id,
            Interaction.item_idx == book.item_idx
        ).first()
        if interaction and (interaction.rating is not None or interaction.comment):
            user_rating = {
                'rating': interaction.rating,
                'comment': interaction.comment
            }

    has_als = ModelStore().has_book_als(book.item_idx)

    return templates.TemplateResponse('book.html', {
        "request": request,
        "book": book_info,
        "user_rating": user_rating,
        'has_als': has_als,
        'has_real_subjects': has_real_subjects,
    })

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
              Interaction.comment != ""
          )
          .scalar() or 0
    )

    q = (
        db.query(Interaction, User)
          .join(User, User.user_id == Interaction.user_id)
          .filter(
              Interaction.item_idx == item_idx,
              Interaction.comment.isnot(None),
              Interaction.comment != ""
          )
    )
    if cursor is not None:
        q = q.filter(Interaction.id < cursor)

    rows = q.order_by(desc(Interaction.id)).limit(limit + 1).all()

    items = []
    for inter, user in rows[:limit]:
        items.append({
            "user_id": user.user_id,
            "username": user.username,
            "rating": inter.rating,
            "comment": inter.comment or "",
            "rated_at": inter.timestamp.isoformat() if inter.timestamp else None,
        })

    has_more = len(rows) > limit
    next_cursor = None
    if has_more:
        last_inter, _ = rows[limit - 1]
        next_cursor = int(last_inter.id)

    return {
        "items": items,
        "total_count": total_count,
        "has_more": has_more,
        "next_cursor": next_cursor,
    }

@router.get("/book/{item_idx}/similar")
def get_similar(item_idx: int, mode: str = "subject", alpha: float = 0.6, top_k: int = 200):
    if mode in ("als", "hybrid") and not ModelStore().has_book_als(item_idx):
        raise HTTPException(
            status_code=422,
            detail="Behavioral similarity is unavailable for this book (no ALS data yet). Try Subject mode."
        )
    
    strategy = get_similarity_strategy(mode=mode, alpha=alpha)
    return strategy.get_similar_books(item_idx, top_k=top_k, alpha=alpha)

# add: mode param with default "auto"
@router.get('/profile/recommend')
async def recommend_for_user(
    user: str = Query(...),
    _id: bool = True,
    top_n: int = 200,
    db: Session = Depends(get_db),
    w: float = 0.6,
    mode: str = Query("auto")  # "auto" | "subject" | "behavioral"
):
    try:
        user_query = db.query(User).options(joinedload(User.favorite_subjects))
        user_obj = user_query.filter(User.user_id == int(user)).first() if _id \
                   else user_query.filter(User.username == user).first()
        if not user_obj:
            raise HTTPException(status_code=404, detail="User not found")

        user_obj.fav_subjects_idxs = [s.subject_idx for s in user_obj.favorite_subjects] or [PAD_IDX]

        user_meta = ModelStore().get_user_meta()
        row = user_meta.loc[user_obj.user_id] if user_obj.user_id in user_meta.index else None
        num_ratings = int(row["user_num_ratings"]) if row is not None else 0

        # choose strategy
        if mode == "behavioral":
            strategy = WarmRecommender()
            return strategy.recommend(user_obj, db=db, top_k=top_n)
        elif mode == "subject":
            strategy = ColdRecommender()
            return strategy.recommend(user_obj, db=db, top_k=top_n, w=w)
        else:
            strategy = RecommenderStrategy.get_strategy(num_ratings) 
            return strategy.recommend(user_obj, db=db, top_k=top_n, w=w)
        
    except Exception as e:
        logger.error(f"Error in /profile/recommend: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/search")
def search_books(
    request: Request,
    query: str = "",
    subjects: Optional[str] = None,
    page: int = Query(0, ge=0),
    page_size: int = Query(60, ge=1, le=200),
    mode: SearchMode = Query(SearchMode.MEILI),
    sort: Optional[str] = Query("bayes_pop:desc", description="e.g. bayes_pop:desc, year:asc"),
    highlight: bool = Query(False),
    crop: Union[bool, int] = Query(False, description="False = no crop, int = crop length"),
    min_score: Optional[float] = Query(None, ge=0, le=1),
    db: Session = Depends(get_db),
):
    search_req = _build_search_request(
        query=query,
        subjects=subjects,
        page=page,
        page_size=page_size,
        mode=mode,
        sort=sort,
        highlight=highlight,
        crop=crop,
        min_score=min_score,
        db=db
    )

    search_response = search_engine.search(search_req)

    # For HTML template
    subject_list = [s.strip() for s in (subjects or "").split(",") if s.strip()]
    subject_suggestions = get_all_subject_counts(db)[:20]

    total_pages = (search_response.total + page_size - 1) // page_size

    return templates.TemplateResponse("search.html", {
        "request": request,
        "results": clean_float_values([r.dict() for r in search_response.results]),
        "query": query,
        "subjects": subject_list,
        "subject_suggestions": subject_suggestions,
        "page": "search",
        "total_results": search_response.total,
        "total_pages": total_pages,
        "current_page": page,
        "has_next": (page + 1) * page_size < search_response.total,
        "has_prev": page > 0,
        "page_size": page_size,
    })


@router.get("/search/json")
def search_books_json(
    query: str = Query(""),
    subjects: Optional[str] = Query(None),
    page: int = Query(0, ge=0),
    page_size: int = Query(60, ge=1, le=200),
    mode: SearchMode = Query(SearchMode.MEILI, description="Currently only 'meili' is supported"),
    sort: Optional[str] = Query("bayes_pop:desc", regex=r"^[\w]+:(asc|desc)$", description="e.g. bayes_pop:desc"),
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
        db=db
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
        }
    }

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

@router.get('/logout')
async def logout():
    response = RedirectResponse(url='/login', status_code=status.HTTP_303_SEE_OTHER)

    response.delete_cookie(key='access_token')

    return response

@router.post("/admin/reload_models")
def reload_models_endpoint(secret: str = Query(...)):
    if secret != os.getenv("ADMIN_SECRET"):
        raise HTTPException(status_code=403)
    try:
        from models.engines_reload import reload_all_models
        reload_all_models()
        return {"status": "reloaded"}
    except Exception as e:
        logger.error("Reload failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")
