from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends
import redis.asyncio as redis

ALLOWED_ORIGINS = [
    "simonbouchard.space",
    "www.simonbouchard.space",
    "recsys.simonbouchard.space",
    "89.117.146.162"

    # Development origins
    # "http://localhost:8000",
    # "http://127.0.0.1:8000",
]

def apply_security(app):
    """
    Apply all production-only middleware, rate limits, CSP headers, etc.
    """

    # --- CORS ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Trusted Hosts ---
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=[
            "simonbouchard.space",
            "www.simonbouchard.space",
            "localhost",
            "recsys.simonbouchard.space",
        ],
    )

    # --- Force HTTPS ---
    app.add_middleware(HTTPSRedirectMiddleware)

    # --- Security Headers ---
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

    # --- Rate Limiter ---
    @app.on_event("startup")
    async def _init_rl():
        r = redis.from_url("redis://127.0.0.1:6379/0")
        await FastAPILimiter.init(r)

def health_dependency():
    return RateLimiter(times=10, seconds=60)
