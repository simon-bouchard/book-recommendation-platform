import jwt
import os
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Cookie, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import hash_password, verify_password
from app.table_models import User, Subject, UserFavSubject

load_dotenv()

router = APIRouter()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


async def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(access_token: str = Cookie(None), db: Session = Depends(get_db)):
    if not access_token:
        return None

    try:
        payload = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        user = db.query(User).filter(User.username == username).first()
        if user:
            return user
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

    return None



class LoginBody(BaseModel):
    username: str
    password: str
    next: str = "/profile"


class SignupBody(BaseModel):
    username: str
    email: str
    password: str
    fav_subjects: list[str] = []


@router.post("/auth/login/json")
async def login_json(body: LoginBody, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == body.username).first()
    if not db_user or not verify_password(body.password, db_user.password):
        return JSONResponse(
            {"error": "Invalid username or password."},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    access_token = await create_access_token({"sub": body.username})
    resp = JSONResponse({"ok": True, "redirect": body.next or "/profile"})
    resp.set_cookie(key="access_token", value=access_token, httponly=True, max_age=3600)
    return resp


@router.post("/auth/signup/json")
async def signup_json(body: SignupBody, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == body.username).first():
        return JSONResponse(
            {"error": "Username already taken. Please choose another."},
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    hashed_pw = hash_password(body.password)
    new_user = User(
        username=body.username,
        email=body.email,
        password=hashed_pw,
        age=None,
        age_group="unknown_age",
        filled_age=False,
        country="Unknown",
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    subject_list = [s.strip() for s in body.fav_subjects if s.strip()]
    if not subject_list:
        subject_list = ["[NO_SUBJECT]"]

    for subject_name in subject_list:
        subject = db.query(Subject).filter(Subject.subject == subject_name).first()
        if not subject:
            subject = Subject(subject=subject_name)
            db.add(subject)
            db.commit()
            db.refresh(subject)
        db.add(UserFavSubject(user_id=new_user.user_id, subject_idx=subject.subject_idx))
    db.commit()

    access_token = await create_access_token({"sub": body.username})
    resp = JSONResponse({"ok": True, "redirect": "/profile"})
    resp.set_cookie(key="access_token", value=access_token, httponly=True, max_age=3600)
    return resp
