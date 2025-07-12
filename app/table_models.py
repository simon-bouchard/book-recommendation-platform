from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base

class Author(Base):
    __tablename__ = 'authors'

    author_id = Column(String(50), primary_key=True, index=True)
    name = Column(String(255), nullable=True)
    birth_date = Column(String(50), nullable=True)
    death_date = Column(String(50), nullable=True)
    bio = Column(Text, nullable=True)
    alternate_names = Column(Text, nullable=True)  # You can store JSON stringified list if needed

    books = relationship('Book', back_populates='author')

class Book(Base):
    __tablename__ = 'books'

    work_id = Column(String(30), primary_key=True, index=True)
    title = Column(Text, nullable=False)
    year = Column(Integer)
    year_bucket = Column(String(50))
    filled_year = Column(Boolean)
    description = Column(Text)
    cover_id = Column(Integer)
    language = Column(String(30))
    num_pages = Column(Integer)
    filled_num_pages = Column(Boolean)
    author_id = Column(String(50), ForeignKey('authors.author_id'))
    isbn = Column(String(20), index=True)

    interactions = relationship('Interaction', back_populates='book')
    subjects = relationship('BookSubject', back_populates='book')
    author = relationship('Author', back_populates='books')

class User(Base):
    __tablename__ = 'users'

    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String(255), unique=True, nullable=False)
    email = Column(String(255), default='user@example.com')
    password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    age = Column(Float)
    age_group = Column(String(50))
    filled_age = Column(Boolean)
    country = Column(String(100))

    interactions = relationship('Interaction', back_populates='user')
    favorite_subjects = relationship('UserFavSubject', back_populates='user')

class Interaction(Base):
    __tablename__ = 'interactions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.user_id'), index=True, nullable=False)
    work_id = Column(String(30), ForeignKey('books.work_id'), index=True, nullable=False)

    rating = Column(Float, nullable=True)               # Null = implicit interaction
    comment = Column(Text, nullable=True)
    timestamp = Column(DateTime, nullable=True)         # Optional: when it occurred
    type = Column(String(50), nullable=True)            # Optional: 'read', 'rated', etc.

    user = relationship("User", back_populates="interactions")
    book = relationship("Book", back_populates="interactions")

class BookSubject(Base):
    __tablename__ = 'book_subjects'

    id = Column(Integer, primary_key=True, autoincrement=True)
    work_id = Column(String(30), ForeignKey('books.work_id'))
    subject = Column(String(255))

    book = relationship('Book', back_populates='subjects')

class UserFavSubject(Base):
    __tablename__ = 'user_fav_subjects'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.user_id'))
    subject = Column(String(255))

    user = relationship('User', back_populates='favorite_subjects')