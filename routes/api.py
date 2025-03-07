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
from models.book_model import reload_model, get_recommendations
from models.user_model import reload_user_model, get_user_recommendations
from datetime import datetime
from sqlalchemy import func
from sqlalchemy.orm import Session
from app.database import SessionLocal, get_db
from app.table_models import Book, User, Rating

router = APIRouter()
templates = Jinja2Templates(directory="templates")

load_dotenv()

@router.get('/login', response_class=HTMLResponse)
def signup_page(request: Request):
    return templates.TemplateResponse('login.html', {'request': request})

@router.get('/profile')
def profile_page(request: Request, current_user: dict = Depends(get_current_user)):
    if not current_user:
        return RedirectResponse(url='/login', status_code=status.HTTP_303_SEE_OTHER)

    user_data = {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "age": current_user.age,
        "location": current_user.location,
    }

    return templates.TemplateResponse('profile.html', {'request': request, 'user': user_data})

@router.post('/rating')
async def new_rating(current_user = Depends(get_current_user), data: dict = Body(...), db: Session = Depends(get_db)):

    if not current_user:
        return RedirectResponse(url='/login', status_code=status.HTTP_303_SEE_OTHER)

    isbn = data.get('isbn')
    rating = data.get('rating')

    if not isbn or not rating:
        raise HTTPException(status_code=400, detail='Missing required fields')

    new_rating = Rating(
        user_id=current_user.id,
        book_isbn=isbn,
        rating=rating,
        comment = data.get('comment'),
    )

    db.merge(new_rating)
    db.commit()
    db.close()

    return {'message': f'Rating successfully submitted/updated!'}

@router.get('/book/{isbn}', response_class=HTMLResponse)
async def book_recommendation(request: Request, isbn: str, current_user: dict = Depends(get_current_user), db: Session = Depends(get_db)): 

    book = db.query(Book).filter(Book.isbn == isbn).first()
    if not book:
        raise HTTPException(status_code=404, detail='Book not found')

    average = db.query(func.avg(Rating.rating)).filter(Rating.book_isbn == isbn).scalar()

    if average: 
        average = round(average, 2)

    if not book: 
        raise HTTPException(status_code=404, detail='Book not found')

    book = {
            'isbn': isbn,
            'title': book.title,
            'author': book.author,
            'genre': book.genre,
            'year': book.year,
            'publisher': book.publisher,
            'average_rating': average
    }

    user_rating = None

    if current_user: 
        user_rating_obj = db.query(Rating).filter(Rating.user_id == current_user.id, Rating.book_isbn == book['isbn']).first()

        if user_rating_obj:
            user_rating={
                'rating': user_rating_obj.rating,
                'comment': user_rating_obj.comment
            }

    return templates.TemplateResponse('book.html', {"request": request, "book": book, 'user_rating': user_rating})

@router.get('/comments')
async def get_comments(book: str = Query(...), isbn: bool = True, limit: int = 5, db: Session = Depends(get_db)):
    if not isbn:
        db_book = db.query(Book).filter(Book.title == book).first() 
        if db_book:
            book = db_book.book_isbn
        else:
            return {'error': 'Book not found'}
        
    comment_query = (
        db.query(Rating.comment, Rating.user_id, User.username)
        .join(User, User.id == Rating.user_id)
        .filter(Rating.book_isbn == book, Rating.comment.isnot(None), Rating.comment != '')
        .limit(limit)
        .all()
    )

    comments = [
            {'comment': comment.comment, 'user_id': comment.user_id, 'username': comment.username}
            for comment in comment_query
    ]

    if not comments:
        raise HTTPException(status_code=404, detail='No comments have been submitted for this book yet')
    return comments

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
async def user_recommendation(user: str, _id: bool = True, n: int = 5):
    if _id:
        if not ObjectId.is_valid(user):
            raise HTTPException(status_code=404, detail='Invalid BSON object format')
        user = ObjectId(user)
    recommendations = await get_user_recommendations(user, _id, n)

    if 'error' in recommendations:
        raise HTTPException(status_code=404, detail=recommendations['error'])
    if recommendations:
        return recommendations
    else:
        raise HTTPException(status_code=404, detail='Unexpected error retrieving recommendations')

@router.get('/logout')
async def logout():
    response = RedirectResponse(url='/login', status_code=status.HTTP_303_SEE_OTHER)

    response.delete_cookie(key='access_token')

    return response

