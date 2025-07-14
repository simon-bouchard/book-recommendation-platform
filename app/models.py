from pydantic import BaseModel, Field, EmailStr
from datetime import datetime
from typing import Optional
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.table_models import BookSubject
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

def get_cached_subject_suggestions(db):
    global _cached_subjects, _last_subject_fetch

    now = time.time()
    if now - _last_subject_fetch > SUBJECT_TTL_SECONDS:
        subq = (
            db.query(
                BookSubject.subject,
                func.count().label("freq")
            )
            .filter(BookSubject.subject.isnot(None))
            .filter(BookSubject.subject != "")
            .group_by(BookSubject.subject)
            .order_by(func.count().desc())
            .limit(100)
            .subquery()
        )
        _cached_subjects = [row.subject for row in db.query(subq.c.subject).all()]
        _last_subject_fetch = now

    return _cached_subjects
