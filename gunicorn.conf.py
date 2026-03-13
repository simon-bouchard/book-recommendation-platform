# gunicorn.conf.py
"""
Gunicorn configuration for the main BookRec FastAPI application.
"""

import os

bind = "127.0.0.1:8000"
workers = 2
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120
graceful_timeout = 120
keepalive = 150

accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info")
