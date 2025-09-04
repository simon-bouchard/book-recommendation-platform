from fastapi import FastAPI, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis

from routes.api import router 
from app.auth import router as auth_router
from routes import chat

import os
from datetime import datetime

app = FastAPI()
"""
ALLOWED_ORIGINS = [
    "simonbouchard.space",
    "www.simonbouchard.space",
    "recsys.simonbouchard.space",
    "89.117.146.162"

    # Development origins
    # "http://localhost:3000",
    # "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],     
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["simonbouchard.space", "www.simonbouchard.space", "localhost", "recsys.simonbouchard.space"]
)

app.add_middleware(HTTPSRedirectMiddleware)

@app.middleware("http")
async def add_security_headers(request, call_next):
    resp = await call_next(request)
    resp.headers["Content-Security-Policy"] = (
		"default-src 'self'; "
        "img-src 'self' data: https:; "
        "script-src 'self' 'unsafe-inline'; "   
        "style-src 'self' 'unsafe-inline'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "frame-ancestors 'none'; "
        "upgrade-insecure-requests"
    )
    resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    resp.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    return resp

@app.on_event("startup")
async def _init_rl():
    r = redis.from_url("redis://127.0.0.1:6379/0")
    await FastAPILimiter.init(r)

@app.get("/api/health", dependencies=[Depends(RateLimiter(times=10, seconds=60))])
async def health():
    return {"ok": True}
"""

app.mount('/static', StaticFiles(directory='static'), name='static')

app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY"))

templates = Jinja2Templates(directory="templates")
templates.env.globals['now'] = datetime.utcnow

app.include_router(router)
app.include_router(chat.router)
app.include_router(auth_router)

@app.get('/', response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse('index.html', {'request': request, 'page': 'home'})


