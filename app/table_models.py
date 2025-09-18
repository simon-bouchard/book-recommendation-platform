from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, DateTime, Text, UniqueConstraint
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
    tones = relationship("BookTone", back_populates="book")
    genre = relationship("BookGenre", back_populates="book", uselist=False)
    vibe  = relationship("BookVibe",  back_populates="book", uselist=False)
    llm_subjects = relationship("BookLLMSubject", back_populates="book")

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

# --- Ontology tables (normalized lookup) ---

class Tone(Base):
    __tablename__ = "tones"
    # Keep IDs identical to ontology CSV (stable integer keys)
    tone_id = Column(Integer, primary_key=True)              # e.g., 1..N
    slug = Column(String(100), unique=True, nullable=False)  # canonical slug
    name = Column(String(200), nullable=True)                # optional display name

    books = relationship("BookTone", back_populates="tone", lazy="dynamic")


class Genre(Base):
    __tablename__ = "genres"
    # Genres are naturally keyed by slug; keep it as PK for 3NF
    slug = Column(String(100), primary_key=True)             # canonical slug
    name = Column(String(200), nullable=True)                # optional display name

    books = relationship("BookGenre", back_populates="genre", lazy="dynamic")


class Vibe(Base):
    __tablename__ = "vibes"
    # Vibes are free text but we still deduplicate to normalize storage
    vibe_id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(Text, unique=True, nullable=False)

    books = relationship("BookVibe", back_populates="vibe", lazy="dynamic")


# --- Link tables (fully normalized; no version/timestamps) ---

class BookTone(Base):
    __tablename__ = "book_tones"
    id = Column(Integer, primary_key=True, autoincrement=True)

    item_idx = Column(Integer, ForeignKey("books.item_idx"), index=True, nullable=False)
    tone_id  = Column(Integer, ForeignKey("tones.tone_id"),  index=True, nullable=False)

    __table_args__ = (UniqueConstraint("item_idx", "tone_id", name="uq_book_tone"),)

    book = relationship("Book", back_populates="tones")
    tone = relationship("Tone", back_populates="books")


class BookGenre(Base):
    __tablename__ = "book_genres"
    # Exactly one genre per book (per current enrichment rules)
    item_idx   = Column(Integer, ForeignKey("books.item_idx"), primary_key=True)
    genre_slug = Column(String(100), ForeignKey("genres.slug"), nullable=False, index=True)

    book  = relationship("Book", back_populates="genre")
    genre = relationship("Genre", back_populates="books")


class BookVibe(Base):
    __tablename__ = "book_vibes"
    # Exactly one vibe per book (short free text, deduped through Vibe)
    item_idx = Column(Integer, ForeignKey("books.item_idx"), primary_key=True)
    vibe_id  = Column(Integer, ForeignKey("vibes.vibe_id"), nullable=False, index=True)

    book = relationship("Book", back_populates="vibe")
    vibe = relationship("Vibe", back_populates="books")


class LLMSubject(Base):
    __tablename__ = "llm_subjects"
    # Canonicalized subject text is stored once and referenced via an integer key
    llm_subject_idx = Column(Integer, primary_key=True, autoincrement=True)
    subject = Column(String(255), unique=True, nullable=False)  # normalized (lowercased/trimmed) phrase

    books = relationship("BookLLMSubject", back_populates="subject", lazy="dynamic")


class BookLLMSubject(Base):
    __tablename__ = "book_llm_subjects"
    id = Column(Integer, primary_key=True, autoincrement=True)

    item_idx = Column(Integer, ForeignKey("books.item_idx"), index=True, nullable=False)
    llm_subject_idx = Column(Integer, ForeignKey("llm_subjects.llm_subject_idx"), index=True, nullable=False)

    __table_args__ = (UniqueConstraint("item_idx", "llm_subject_idx", name="uq_book_llm_subject"),)

    book = relationship("Book", back_populates="llm_subjects")
    subject = relationship("LLMSubject", back_populates="books")