from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')

Base = declarative_base()

if DATABASE_URL:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,        # check connection health before use
        pool_recycle=3600,         # recycle conns older than 1 hour
        pool_size=10,              # adjust for your expected load
        max_overflow=20,           # allow bursts
        pool_timeout=30,           # wait time for a connection
        connect_args={"connect_timeout": 10}
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
else:
    engine = None
    SessionLocal = None

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

