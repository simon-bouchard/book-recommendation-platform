from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base

class Author(Base):
    __tablename__ = 'authors'

    author_idx = Column(Integer, primary_key=True, autoincrement=True)
    external_id = Column(String(50), unique=True, index=True)  # optional
    name = Column(String(255), nullable=True)
    birth_date = Column(String(50), nullable=True)
    death_date = Column(String(50), nullable=True)
    bio = Column(Text, nullable=True)
    alternate_names = Column(Text, nullable=True)

    books = relationship('Book', back_populates='author')

class Book(Base):
    __tablename__ = 'books'

    item_idx = Column(Integer, primary_key=True, index=True)
    work_id = Column(String(30), index=True)
    title = Column(Text, nullable=False)
    year = Column(Integer)
    year_bucket = Column(String(50))
    filled_year = Column(Boolean)
    description = Column(Text)
    cover_id = Column(Integer)
    language = Column(String(30))
    num_pages = Column(Integer)
    filled_num_pages = Column(Boolean)
    author_idx = Column(Integer, ForeignKey('authors.author_idx'), nullable=True)
    isbn = Column(String(20), index=True)
    main_subject = Column(String(255), nullable=True)

    interactions = relationship('Interaction', back_populates='book')
    subjects = relationship('BookSubject', back_populates='book')
    author = relationship('Author', back_populates='books')

class Subject(Base):
    __tablename__ = 'subjects'

    subject_idx = Column(Integer, primary_key=True, index=True)
    subject = Column(String(255), unique=True, nullable=False)
    
    books = relationship('BookSubject', back_populates='subject', lazy='dynamic')
    favorited_by = relationship('UserFavSubject', back_populates='subject', lazy='dynamic')

class BookSubject(Base):
    __tablename__ = 'book_subjects'

    id = Column(Integer, primary_key=True, autoincrement=True)
    item_idx = Column(Integer, ForeignKey('books.item_idx'), index=True, nullable=False)
    subject_idx = Column(Integer, ForeignKey('subjects.subject_idx'), index=True, nullable=False)

    book = relationship('Book', back_populates='subjects')
    subject = relationship('Subject', back_populates='books')

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

class UserFavSubject(Base):
    __tablename__ = 'user_fav_subjects'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.user_id'), index=True, nullable=False)
    subject_idx = Column(Integer, ForeignKey('subjects.subject_idx'), index=True, nullable=False)

    user = relationship('User', back_populates='favorite_subjects')
    subject = relationship('Subject', back_populates='favorited_by')

class Interaction(Base):
    __tablename__ = 'interactions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.user_id'), index=True, nullable=False)
    item_idx = Column(Integer, ForeignKey('books.item_idx'), index=True, nullable=False)

    rating = Column(Float, nullable=True)
    comment = Column(Text, nullable=True)
    timestamp = Column(DateTime, nullable=True)
    type = Column(String(50), nullable=True)

    user = relationship("User", back_populates="interactions")
    book = relationship("Book", back_populates="interactions")