from sqlalchemy import (
    Column, Integer, String, Float, Boolean, ForeignKey, DateTime, Text, UniqueConstraint, JSON, ForeignKeyConstraint
)
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base


# =========================
# Core (existing)
# =========================

class Author(Base):
    __tablename__ = 'authors'

    author_idx = Column(Integer, primary_key=True, autoincrement=True)
    external_id = Column(String(50), unique=True, index=True)
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

    # Enrichment relations
    tones = relationship('BookTone', back_populates='book', cascade="all, delete-orphan", passive_deletes=True)
    genre = relationship('BookGenre', back_populates='book', cascade="all, delete-orphan", passive_deletes=True)
    vibe  = relationship('BookVibe',  back_populates='book', cascade="all, delete-orphan", passive_deletes=True)
    llm_subjects = relationship('BookLLMSubject', back_populates='book', cascade="all, delete-orphan", passive_deletes=True)
    ol_subjects = relationship('BookOLSubject', back_populates='book')


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


# =========================
# Enrichment: Ontologies
# =========================

class Tone(Base):
    __tablename__ = "tones"
    
    tone_id = Column(Integer, primary_key=True)
    slug = Column(String(100), nullable=False)
    name = Column(String(200), nullable=True)
    ontology_version = Column(String(32), nullable=False, default='v1', index=True)
    
    __table_args__ = (
        UniqueConstraint("slug", "ontology_version", name="uq_tone_slug_version"),
    )
    
    books = relationship("BookTone", back_populates="tone", lazy="dynamic")

class Genre(Base):
    __tablename__ = "genres"
    
    slug = Column(String(100), nullable=False)
    name = Column(String(200), nullable=True)
    ontology_version = Column(String(32), nullable=False, default='v1')
    
    __table_args__ = (
        {'extend_existing': True},
    )
    
    # Composite primary key
    __mapper_args__ = {'primary_key': [slug, ontology_version]}
    
    books = relationship("BookGenre", back_populates="genre", lazy="dynamic")


class Vibe(Base):
    __tablename__ = "vibes"
    vibe_id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(String(512), unique=True, nullable=False)

    books = relationship("BookVibe", back_populates="vibe", lazy="dynamic")


# =========================
# Enrichment: LLM Subjects
# =========================

class LLMSubject(Base):
    __tablename__ = "llm_subjects"
    llm_subject_idx = Column(Integer, primary_key=True, autoincrement=True)
    subject = Column(String(255), unique=True, nullable=False)

    books = relationship("BookLLMSubject", back_populates="subject", lazy="dynamic")


# =========================
# Enrichment: Link Tables (UPDATED WITH tags_version)
# =========================

class BookTone(Base):
    __tablename__ = "book_tones"

    id = Column(Integer, primary_key=True, autoincrement=True)
    item_idx = Column(Integer, ForeignKey("books.item_idx", onupdate="CASCADE", ondelete="CASCADE"), index=True, nullable=False)
    tone_id = Column(Integer, ForeignKey("tones.tone_id", onupdate="CASCADE", ondelete="RESTRICT"), index=True, nullable=False)
    tags_version = Column(String(32), nullable=False, default='v1', index=True)

    __table_args__ = (UniqueConstraint("item_idx", "tone_id", "tags_version", name="uq_book_tone_version"),)

    book = relationship("Book", back_populates="tones")
    tone = relationship("Tone", back_populates="books")


class BookGenre(Base):
    __tablename__ = "book_genres"

    item_idx = Column(Integer, ForeignKey("books.item_idx", onupdate="CASCADE", ondelete="CASCADE"), nullable=False)
    genre_slug = Column(String(100), nullable=False)
    genre_ontology_version = Column(String(32), nullable=False, default='v1')
    tags_version = Column(String(32), nullable=False, default='v1', index=True)

    __table_args__ = (
        ForeignKeyConstraint(
            ['genre_slug', 'genre_ontology_version'],
            ['genres.slug', 'genres.ontology_version'],
            onupdate="CASCADE",
            ondelete="RESTRICT"
        ),
        {'extend_existing': True},
    )
    
    # Composite primary key
    __mapper_args__ = {'primary_key': [item_idx, tags_version]}

    book = relationship("Book", back_populates="genre")
    genre = relationship("Genre", back_populates="books")


class BookVibe(Base):
    __tablename__ = "book_vibes"

    item_idx = Column(Integer, ForeignKey("books.item_idx", onupdate="CASCADE", ondelete="CASCADE"), nullable=False)
    vibe_id = Column(Integer, ForeignKey("vibes.vibe_id", onupdate="CASCADE", ondelete="RESTRICT"), nullable=False, index=True)
    tags_version = Column(String(32), nullable=False, default='v1', index=True)

    __table_args__ = (
        {'extend_existing': True},
    )
    
    # Composite primary key
    __mapper_args__ = {'primary_key': [item_idx, tags_version]}

    book = relationship("Book", back_populates="vibe")
    vibe = relationship("Vibe", back_populates="books")


class BookLLMSubject(Base):
    __tablename__ = "book_llm_subjects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    item_idx = Column(Integer, ForeignKey("books.item_idx", onupdate="CASCADE", ondelete="CASCADE"), index=True, nullable=False)
    llm_subject_idx = Column(Integer, ForeignKey("llm_subjects.llm_subject_idx", onupdate="CASCADE", ondelete="RESTRICT"), index=True, nullable=False)
    tags_version = Column(String(32), nullable=False, default='v1', index=True)

    __table_args__ = (UniqueConstraint("item_idx", "llm_subject_idx", "tags_version", name="uq_book_llm_subject_version"),)

    book = relationship("Book", back_populates="llm_subjects")
    subject = relationship("LLMSubject", back_populates="books")


# =========================
# Enrichment Errors
# =========================

class EnrichmentError(Base):
    __tablename__ = "enrichment_errors"

    # Composite PK: one error per book per version
    item_idx = Column(Integer, nullable=False)
    tags_version = Column(String(32), nullable=False)
    
    __table_args__ = (
        {'extend_existing': True},
    )
    __mapper_args__ = {'primary_key': [item_idx, tags_version]}
    
    # Temporal tracking
    first_seen_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_seen_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    occurrence_count = Column(Integer, nullable=False, default=1)
    
    # Error details
    stage = Column(String(64), nullable=False, index=True)
    error_code = Column(String(64), nullable=False, index=True)
    error_field = Column(String(128), nullable=True)
    error_msg = Column(Text, nullable=False)
    
    # Book context
    title = Column(String(256), nullable=True)
    author = Column(String(256), nullable=True)
    attempted = Column(JSON, nullable=True)
    
    # NEW: Run tracking
    last_run_id = Column(String(64), nullable=True, index=True)  # Which run it last failed in
    run_history = Column(JSON, nullable=True)  # Array of {run_id, timestamp} if you want full history
    
    book = relationship("Book", viewonly=True)

# =========================
# Staging Tables (UPDATED WITH tags_version)
# =========================

class TmpEnrichmentErrorsLoad(Base):
    __tablename__ = "tmp_enrichment_errors_load"
    
    item_idx = Column(Integer, primary_key=True)
    first_seen_at = Column(DateTime, nullable=True)
    last_seen_at = Column(DateTime, nullable=True)
    occurrence_count = Column(Integer, default=1)
    stage = Column(String(64), nullable=True)
    error_code = Column(String(64), nullable=True)
    error_field = Column(String(128), nullable=True)
    error_msg = Column(Text, nullable=True)
    tags_version = Column(String(32), nullable=True)
    title = Column(String(256), nullable=True)
    author = Column(String(256), nullable=True)
    attempted = Column(JSON, nullable=True)

class OLSubject(Base):
    __tablename__ = 'ol_subjects'

    ol_subject_idx = Column(Integer, primary_key=True, autoincrement=True)
    subject = Column(String(500), unique=True, nullable=False)  # VARCHAR(500) for MySQL compatibility

    books = relationship('BookOLSubject', back_populates='ol_subject', lazy='dynamic')


class BookOLSubject(Base):
    __tablename__ = 'book_ol_subjects'

    id = Column(Integer, primary_key=True, autoincrement=True)
    item_idx = Column(Integer, ForeignKey('books.item_idx'), index=True, nullable=False)
    ol_subject_idx = Column(Integer, ForeignKey('ol_subjects.ol_subject_idx'), index=True, nullable=False)

    __table_args__ = (UniqueConstraint('item_idx', 'ol_subject_idx', name='uq_book_ol_subject'),)

    book = relationship('Book', back_populates='ol_subjects')
    ol_subject = relationship('OLSubject', back_populates='books')
