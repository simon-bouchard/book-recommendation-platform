from fastapi import FastAPI, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from starlette.middleware.sessions import SessionMiddleware

from routes.api import router 
from routes.auth import router as auth_router
from routes.chat import router as chat_router

import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

SECURE_MODE = os.getenv("SECURE_MODE", "true").lower() == "true"

def _no_limit():
        return None

if SECURE_MODE:
    from security_settings import apply_security, health_dependency
    apply_security(app)
    limiter_dep = health_dependency()

else:
    # No rate limiting or security
    limiter_dep = _no_limit

@app.get("/health", dependencies=[Depends(limiter_dep)])
def health():
        return {"status": "ok"}

app.mount('/static', StaticFiles(directory='static'), name='static')

app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY"))

templates = Jinja2Templates(directory="templates")
templates.env.globals['now'] = datetime.utcnow

app.include_router(router)
app.include_router(chat_router)
app.include_router(auth_router)

@app.get('/', response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse('index.html', {'request': request, 'page': 'home'})


