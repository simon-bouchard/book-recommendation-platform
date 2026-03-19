import jwt
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Form, Request, status, Cookie
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from app.models import hash_password, verify_password
import os
from dotenv import load_dotenv
from datetime import datetime
from sqlalchemy.orm import Session
from app.database import get_db
from app.table_models import User, Subject, UserFavSubject
from datetime import datetime

load_dotenv()

router = APIRouter()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

templates = Jinja2Templates(directory="templates")
templates.env.globals["now"] = datetime.utcnow


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


@router.post("/auth/signup")
async def signup(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    fav_subjects: str = Form(""),
    db: Session = Depends(get_db),
):
    existing_user = db.query(User).filter(User.username == username).first()

    if existing_user:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "page": "login",
                "error": "Username already exists. Please choose another.",
            },
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    final_age = None
    filled_age = False
    age_group = "unknown_age"
    country_name = "Unknown"

    hashed_pw = hash_password(password)

    new_user = User(
        username=username,
        email=email,
        password=hashed_pw,
        age=final_age,
        age_group=age_group,
        filled_age=filled_age,
        country=country_name,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    subject_list = [s.strip() for s in fav_subjects.split(",") if s.strip()]
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

    access_token = await create_access_token({"sub": username})
    response = RedirectResponse(url="/profile", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(key="access_token", value=access_token, httponly=True, max_age=3600)
    request.session["flash_success"] = "Profile successfully created."
    return response


@router.post("/auth/login", response_class=HTMLResponse)
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    db_user = db.query(User).filter(User.username == username).first()

    # On any auth failure, re-render login.html with a friendly message
    if not db_user or not verify_password(password, db_user.password):
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "page": "login", "error": "Invalid username or password."},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    # Success → set cookie and redirect to profile
    access_token = await create_access_token({"sub": username})
    response = RedirectResponse(url="/profile", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(key="access_token", value=access_token, httponly=True, max_age=3600)
    return response
