import jwt
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, Form, Request, status, Cookie, Response
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from motor.motor_asyncio import AsyncIOMotorClient
from app.models import UserSignup, UserLogin, hash_password, verify_password
import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from sqlalchemy.orm import Session
from app.database import SessionLocal, get_db
from app.table_models import User

load_dotenv()

router = APIRouter()

SECRET_KEY = os.getenv('SECRET_KEY')
ALGORITHM = 'HS256'
ACCESS_TOKEN_EXPIRE_MINUTES = 60
oauth2_scheme = OAuth2PasswordBearer(tokenUrl='token')

templates = Jinja2Templates(directory='../templates')

async def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({'exp': expire})
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

@router.post('/auth/signup')
async def signup(request: Request, username: str = Form(...), email: str = Form(...), password: str = Form(...), age: int = Form(...), location: str = Form(...), db: Session=Depends(get_db)):
    existing_user = db.query(User).filter(User.username == username).first() #await users.find_one({'username': username})

    if existing_user:
        raise HTTPException(status_code=400, detail='Username already taken')

    hashed_password = hash_password(password)
    new_user = User(username=username, email=email, password=hashed_password, age=age, location=location)

    db.add(new_user)
    db.commit()
    db.close()

    access_token = await create_access_token({'sub': username})

    response = RedirectResponse(url="/profile", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(key="access_token", value=access_token, httponly=True)
    return response

@router.post('/auth/login', response_class=HTMLResponse)
async def login(request: Request, username: str = Form(...), password: str = Form(...), db: Session=Depends(get_db)):

    db_user = db.query(User).filter(User.username == username).first()

    if not db_user:
        raise HTTPException(status_code=401, detail='Invalid username')
    if not verify_password(password, db_user.password):
        raise HTTPException(status_code=401, detail='Invalid password')

    access_token = await create_access_token({'sub': username})

    response = RedirectResponse(url="/profile", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(key="access_token", value=access_token, httponly=True, max_age=3600)

    return response
