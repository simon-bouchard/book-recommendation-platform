import pandas as pd
from fastapi import APIRouter, HTTPException, Form, Request, FastAPI, Depends, status, Body, Query, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv
import jwt
from datetime import datetime, timedelta
from bson import ObjectId
import pickle

from datetime import datetime
from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from app.auth import get_current_user
from app.database import SessionLocal, get_db
from app.table_models import Book, User, Interaction, BookSubject, Subject, UserFavSubject
from app.models import get_all_subject_counts
#from models.knn_utils import get_similar_books, set_book_meta
from models.cold_user_recs import recommend_books_for_cold_user

import logging
import pycountry
from typing import List, Optional

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="templates")

load_dotenv()

@router.get('/login', response_class=HTMLResponse)
def signup_page(request: Request):
    countries = sorted([c.name for c in pycountry.countries])
    return templates.TemplateResponse('login.html', {
        'request': request,
        'countries': countries
    })

@router.get('/profile')
def profile_page(request: Request, current_user: dict = Depends(get_current_user)):
    if not current_user:
        return RedirectResponse(url='/login', status_code=status.HTTP_303_SEE_OTHER)

    user_data = {
        "id": current_user.user_id,
        "username": current_user.username,
        "email": current_user.email,
        "age": current_user.age,
        "country": current_user.country,
        "favorite_subjects": [
            s.subject.subject for s in current_user.favorite_subjects
            if s.subject and s.subject.subject != "[NO_SUBJECT]"
        ]
    }

    return templates.TemplateResponse('profile.html', {'request': request, 'user': user_data})

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

    return {'message': 'Interaction recorded successfully'}

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
        'subjects': subjects
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

    return templates.TemplateResponse('book.html', {
        "request": request,
        "book": book_info,
        "user_rating": user_rating
    })

@router.get('/comments')
async def get_comments(book: str = Query(...), isbn: bool = False, limit: int = 5, db: Session = Depends(get_db)):
    # Resolve work_id
    if isbn:
        db_book = db.query(Book).filter(Book.isbn == book).first()
        if not db_book:
            raise HTTPException(status_code=404, detail='Book not found')
        item_idx = db_book.item_idx
    else:
        item_idx = book

    # Query interactions with comments
    comment_query = (
        db.query(Interaction.comment, Interaction.user_id, User.username, Interaction.rating)
        .join(User, User.user_id == Interaction.user_id)
        .filter(
            Interaction.item_idx == item_idx,
            Interaction.comment.isnot(None),
            Interaction.comment != ''
        )
        .limit(limit)
        .all()
    )

    if not comment_query:
        raise HTTPException(status_code=404, detail='No comments have been submitted for this book yet')

    return [
        {
            'comment': row.comment,
            'user_id': row.user_id,
            'username': row.username,
            'rating': row.rating
        }
        for row in comment_query
    ]


@router.get("/book/{item_idx}/similar")
def get_similar(item_idx: int, db: Session = Depends(get_db)):
    # Get top-k similar item_idxs
    results = get_similar_books(item_idx, top_k=10, method="faiss")  # metadata-free

    similar_ids = [r["item_idx"] for r in results]

    # Fetch metadata for just those books
    books = (
        db.query(Book)
        .filter(Book.item_idx.in_(similar_ids))
        .options(joinedload(Book.subjects).joinedload(BookSubject.subject))
        .all()
    )

    meta = {
        b.item_idx: {
            "title": b.title,
            "cover_id": b.cover_id,
            "author": b.author.name if b.author else None,
            "year": b.year,
            "isbn": b.isbn,
        }
        for b in books
    }

    # Attach metadata to results
    for r in results:
        m = meta.get(r["item_idx"], {})
        r["title"] = m.get("title", "[Unknown]")
        r["cover_id"] = m.get("cover_id")
        r["author"] = m.get("author")
        r["year"] = m.get("year")
        r["isbn"] = m.get("isbn")

    return results

@router.get('/profile/recommend')
async def recommend_for_user(
    user: str = Query(...),
    _id: bool = True,
    top_n: int = 100,
    db: Session = Depends(get_db)
):
    try:
        # Resolve user_id
        if _id:
            user_id = int(user)
        else:
            user_obj = db.query(User).filter(User.username == user).first()
            if not user_obj:
                raise HTTPException(status_code=404, detail="User not found")
            user_id = user_obj.user_id

        # Count valid ratings
        num_ratings = db.query(Interaction).filter(
            Interaction.user_id == user_id,
            Interaction.rating.isnot(None)
        ).count()

        # Cold-start logic
        if num_ratings < 10:
            return recommend_books_for_cold_user(user_id=user_id, top_k=top_n)

        # Warm-start fallback (not implemented here)
        raise HTTPException(status_code=501, detail="Warm-start not implemented yet")

    except Exception as e:
        logger.error(f"Error in /profile/recommend: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/search")
def search_books(request: Request, query: str = "", subjects: Optional[str] = Query(default=None), db: Session = Depends(get_db)):
    subject_idxs = []
    if subjects:
        subject_list = [s.strip() for s in subjects.split(",") if s.strip()]
        subject_rows = db.query(Subject).filter(Subject.subject.in_(subject_list)).all()
        subject_idxs = [s.subject_idx for s in subject_rows]

    q = db.query(Book).join(BookSubject, Book.item_idx == BookSubject.item_idx).filter(Book.title.ilike(f"%{query}%"))

    if subject_idxs:
        q = (
            q.filter(BookSubject.subject_idx.in_(subject_idxs))
            .group_by(Book.item_idx)
            .having(func.count(func.distinct(BookSubject.subject_idx)) == len(subject_idxs))
        )
    else:
        q = q.group_by(Book.item_idx)

    results = q.limit(100).all()

    subject_suggestions = get_all_subject_counts(db)

    return templates.TemplateResponse("search.html", {
        "request": request,
        "results": results,
        "query": query,
        "subjects": subjects or [],
        "subject_suggestions": subject_suggestions[:20],
    })

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

