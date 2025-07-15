from pydantic import BaseModel, Field, EmailStr
from datetime import datetime
from typing import Optional
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.table_models import BookSubject, Subject
import time

_cached_subjects = []
_last_subject_fetch = 0
SUBJECT_TTL_SECONDS = 1000 


pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

class UserSignup(BaseModel):
    location: str 
    username: str 
    email: EmailStr = Field(default='unknown@example.com')
    created_at: datetime = Field(default_factory=datetime.utcnow)
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class Rating(BaseModel):
    user_id: str
    isbn: str
    rating: int = Field(ge=0, le=10)
    timestamp: datetime = Field(dafault_factory=datetime.utcnow)

class Book(BaseModel):
    id: str = Field(..., alias='_id')

def get_cached_subject_suggestions(db: Session):
    # Join BookSubject → Subject, count frequency of each subject string
    subject_counts = (
        db.query(Subject.subject, func.count(Subject.subject_idx).label("count"))
        .join(BookSubject, Subject.subject_idx == BookSubject.subject_idx)
        .group_by(Subject.subject)
        .order_by(func.count(Subject.subject_idx).desc())
        .limit(20)
        .all()
    )

    return [s.subject for s in subject_counts]