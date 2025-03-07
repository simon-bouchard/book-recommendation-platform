import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv
from app.table_models import Book, User, Rating
from app.database import get_db, SessionLocal
from sqlalchemy.orm import Session
from fastapi import Depends
from sqlalchemy.orm import Session
from app.database import SessionLocal, get_db
from app.table_models import Book, Rating

user_book_matrix = None
sparse_matrix = None
book_model = None
isbn_to_index = None

load_dotenv()

async def reload_model():
    print('Model reload...')
    global user_book_matrix, sparse_matrix, book_model, isbn_to_index

    db = SessionLocal()

    try:
        ratings_query = db.query(Rating.user_id, Rating.book_isbn, Rating.rating).all()
        df_ratings = pd.DataFrame(ratings_query, columns=['user_id', 'isbn', 'rating'])

        if df_ratings.empty:
            return

        user_count = df_ratings['user_id'].value_counts()
        valid_users = user_count[user_count >= 200].index

        book_count = df_ratings['isbn'].value_counts()
        valid_books = book_count[book_count >= 100].index

        df_ratings = df_ratings[(df_ratings['user_id'].isin(valid_users)) & (df_ratings['isbn'].isin(valid_books))]

        df_ratings['rating'] = df_ratings['rating'].astype(int)

        user_book_matrix = df_ratings.pivot(index='isbn', columns='user_id', values='rating').fillna(0)
        sparse_matrix = csr_matrix(user_book_matrix.values)

        book_model = NearestNeighbors(metric='cosine', algorithm='brute')
        book_model.fit(sparse_matrix)

        isbn_to_index = {isbn: idx for idx, isbn in enumerate(user_book_matrix.index)}

        print('Model reloaded')

    finally:
        db.close()

async def get_recommendations(book: str, isbn: bool = True, db: Session = None):

    if db is None:
        db = SessionLocal()

    if isbn:
        book_isbn = book.strip()
    else:
        book_entry = db.query(Book).filter(Book.title == book).first() 
        book_isbn = book_entry.isbn

    if not book_isbn:
        return { 'error': "Book not found"}
    print(f"Processed book_isbn: {book_isbn}")

    if book_model is None or user_book_matrix is None or isbn_to_index is None:
        await reload_model()
    print(list(isbn_to_index.keys())[:5])
    
    if str(book_isbn) not in isbn_to_index:
        print(f"ISBN {book_isbn} not found in isbn_to_index. Available ISBNs: {list(isbn_to_index.keys())[:5]}")
        return { 'error': "Book not found ** (books with less than 100 ratings can't get recommendations)"}

    query_index = isbn_to_index[book_isbn]

    distances, indices = book_model.kneighbors(sparse_matrix[query_index], n_neighbors=6)

    recommendations = []

    for idx, dist in zip(indices[0][-1:0:-1], distances[0][-1:0:-1]):
        neighbor_isbn = user_book_matrix.index[idx]
        book_result = db.query(Book).filter(Book.isbn == neighbor_isbn).first()
        if book_result: 
            recommendations.append({'title': book_result.title, 'similarity': round(float(dist), 2)})
        
    return recommendations

