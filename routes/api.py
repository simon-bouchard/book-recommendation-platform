import pandas as pd
from fastapi import APIRouter, HTTPException, Form, Request, FastAPI, Depends, status, Body, Query, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv
import jwt
from datetime import datetime, timedelta
from app.auth import get_current_user
from bson import ObjectId
#from models.book_model import reload_model, get_recommendations
#from models.user_model import reload_user_model, get_user_recommendations
from datetime import datetime
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from app.database import SessionLocal, get_db
from app.table_models import Book, User, Interaction, BookSubject
from app.models import get_cached_subject_suggestions
import logging
import pycountry
from typing import List

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
    }

    return templates.TemplateResponse('profile.html', {'request': request, 'user': user_data})

@router.post('/rating')
async def new_rating(current_user = Depends(get_current_user), data: dict = Body(...), db: Session = Depends(get_db)):

    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    work_id = data.get('work_id')
    rating = data.get('rating')  # Can be None for 'read'
    comment = data.get('comment')

    book = db.query(Book).filter(Book.work_id == work_id).first()
    if not book:
        raise HTTPException(status_code=404, detail='Book not found')

    interaction_type = 'rated' if rating is not None else 'read'
    now = datetime.utcnow()

    interaction = db.query(Interaction).filter(
        Interaction.user_id == current_user.user_id,
        Interaction.work_id == book.work_id
    ).first()

    if interaction:
        interaction.rating = rating
        interaction.comment = comment
        interaction.type = interaction_type
        interaction.timestamp = now
    else:
        interaction = Interaction(
            user_id=current_user.user_id,
            work_id=book.work_id,
            rating=rating,
            comment=comment,
            type=interaction_type,
            timestamp=now
        )
        db.add(interaction)

    db.commit()
    db.close()

    return {'message': 'Interaction recorded successfully'}

@router.get('/book/{work_id}', response_class=HTMLResponse)
async def book_recommendation(request: Request, work_id: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):

    book = db.query(Book).filter(Book.work_id == work_id).first()   
    if not book:
        raise HTTPException(status_code=404, detail='Book not found')

    # Average rating
    average = db.query(func.avg(Interaction.rating)).filter(
        Interaction.work_id == book.work_id,
        Interaction.rating.isnot(None)
    ).scalar()

    # Count of ratings
    rating_count = db.query(Interaction).filter(
        Interaction.work_id == book.work_id,
        Interaction.rating.isnot(None)
    ).count()

    # Subjects list
    subjects = [s.subject for s in book.subjects]

    book_info = {
        'isbn': book.isbn,
        'work_id': book.work_id,
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
            Interaction.work_id == book.work_id
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
        work_id = db_book.work_id
    else:
        work_id = book

    # Query interactions with comments
    comment_query = (
        db.query(Interaction.comment, Interaction.user_id, User.username, Interaction.rating)
        .join(User, User.user_id == Interaction.user_id)
        .filter(
            Interaction.work_id == work_id,
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


@router.get('/recommend')
async def recommend_books(book: str = Query(...), isbn: bool = True):
    recommendations = await get_recommendations(book, isbn)

    if 'error' in recommendations:
        raise HTTPException(status_code=404, detail=recommendations['error'])
    if recommendations:
        return recommendations
    else:
        raise HTTPException(status_code=404, detail="Book not found (books with less than 100 ratings can't have recommendations.)")

@router.get('/profile/recommend')
async def user_recommendation(user: str = Query(...), _id: bool = True, top_n: int = 5):

    try:
        recommendations = await get_user_recommendations(user, _id, top_n)

    except ValueError:  
        raise HTTPException(status_code=200, detail="User doesn't have recommendations.")
    except HTTPException as e:  
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in /profile/recommend: {str(e)}")
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again later.")

    return recommendations

@router.get('/search', response_class=HTMLResponse)
def search_books(
    request: Request,
    query: str = Query(None),
    subjects: List[str] = Query([]),
    db: Session = Depends(get_db)
):
    subjects = [s.strip() for s in subjects if s.strip()]
    subjects = list(dict.fromkeys(subjects))[:3]

    subject_suggestions = get_cached_subject_suggestions(db)
    results = []

    if query:
        if subjects:
            q = (
                db.query(Book)
                .join(BookSubject)
                .filter(Book.title.ilike(f"%{query}%"))
                .filter(BookSubject.subject.in_(subjects))
                .group_by(Book.work_id)
                .having(func.count(BookSubject.subject) >= len(subjects))
            )
        else:
            q = db.query(Book).filter(Book.title.ilike(f"%{query}%"))

        results = q.limit(50).all()


    return templates.TemplateResponse('search.html', {
        'request': request,
        'query': query,
        'subjects': subjects,
        'subject_suggestions': subject_suggestions,
        'results': results
    })

@router.get('/logout')
async def logout():
    response = RedirectResponse(url='/login', status_code=status.HTTP_303_SEE_OTHER)

    response.delete_cookie(key='access_token')

    return response

