from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from starlette.middleware.sessions import SessionMiddleware
from routes.api import router 
from app.auth import router as auth_router
import os
from datetime import datetime

app = FastAPI()

app.mount('/static', StaticFiles(directory='static'), name='static')

app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY"))

templates = Jinja2Templates(directory="templates")
templates.env.globals['now'] = datetime.utcnow

app.include_router(router)
app.include_router(auth_router)

@app.get('/', response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse('index.html', {'request': request, 'page': 'home'})


