from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from app.database import Base
from datetime import datetime

class Book(Base):
    __tablename__ = 'books'

    isbn = Column(String(13), primary_key=True, index=True)
    title = Column(String(255), index=True)
    author = Column(String(255))
    genre = Column(String(255))
    year = Column(Integer)
    publisher = Column(String(255))
    image_url_s = Column(String(255))
    image_url_m = Column(String(255))
    image_url_l = Column(String(255))

    ratings = relationship('Rating', back_populates='book')

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    username = Column(String(255), default='test_user')
    email= Column(String(255), default='test_user@example.com')
    password = Column(String(255), default='test')
    created_at = Column(DateTime, default=datetime.utcnow)
    age = Column(Integer)
    country = Column(String(255))
    location = Column(String(255))
    
    ratings = relationship('Rating', back_populates='user')

class Rating(Base):
    __tablename__ = 'ratings'

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    book_isbn = Column(String(13), ForeignKey('books.isbn'))
    rating = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    comment = Column(String(255))

    user = relationship('User', back_populates='ratings')
    book = relationship('Book', back_populates='ratings')
