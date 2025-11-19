# app/search/search_utils.py

from typing import Optional, Union
from sqlalchemy.orm import Session
from sqlalchemy import func
from threading import Thread
from meilisearch import Client
import os
from dotenv import load_dotenv
import logging

from app.table_models import Subject, Interaction
from app.search.engine import SearchEngine
from app.search.models import SearchRequest, SearchMode
from app.database import SessionLocal

# Setup logging
logger = logging.getLogger(__name__)

search_engine = SearchEngine()

load_dotenv()

# Meilisearch client setup
client = Client("http://localhost:7700", os.getenv("MEILI_MASTER_KEY"))
index = client.index("books")


def get_search_results(query, subject, page, page_size, db):
    filters = {}
    if subject:
        filters["subject_ids"] = subject

    search_request = SearchRequest(
        query=query,
        mode=SearchMode.MEILI,
        sort="bayes_pop:desc", 
        filters=filters,
        page=page,
        page_size=page_size + 1
    )

    search_response = search_engine.search(search_request)
    results = search_response.results
    total = search_response.total
    
    has_next = len(results) > page_size
    results = results[:page_size]
    
    return results, has_next, total


def _build_search_request(
    query: str,
    subjects: Optional[str],
    page: int,
    page_size: int,
    mode: SearchMode,
    sort: Optional[str],
    highlight: bool,
    crop: Union[bool, int],
    min_score: Optional[float],
    db: Session,
) -> SearchRequest:
    filters = {}
    if subjects:
        subject_list = [s.strip() for s in subjects.split(",") if s.strip()]
        subject_rows = (
            db.query(Subject.subject_idx)
            .filter(Subject.subject.in_(subject_list))
            .all()
        )
        subject_idxs = [row.subject_idx for row in subject_rows]
        if subject_idxs:
            filters["subject_ids"] = subject_idxs  # Meili expects subject_ids array

    return SearchRequest(
        query=query,
        mode=mode,
        filters=filters,
        sort=sort,
        page=page,
        page_size=page_size,
        highlight=highlight,
        crop=crop if isinstance(crop, int) or crop is False else 25,
        min_score=min_score,
        attributes_to_retrieve=[
            "item_idx", "title", "author", "cover_id", "year", "isbn", "description"
        ],  # optional: be explicit
    )


def update_book_ratings_in_meili(book_id: int):
    """
    Unified function to update a book's ratings in Meilisearch.
    Can be called from any endpoint that affects ratings.
    """
    def _update():
        try:
            # Create a new database session for the background thread
            db = SessionLocal()
            
            # Calculate current ratings for the book
            avg_result = db.query(func.avg(Interaction.rating)).filter(
                Interaction.item_idx == book_id,
                Interaction.rating.isnot(None)
            ).scalar()
            
            count_result = db.query(func.count(Interaction.rating)).filter(
                Interaction.item_idx == book_id,
                Interaction.rating.isnot(None)
            ).scalar()

            avg_rating = round(avg_result, 3) if avg_result else 0.0
            num_ratings = count_result or 0

            index.update_documents([{
                "item_idx": book_id,
                "num_ratings": num_ratings,
                "avg_rating": avg_rating
                # bayes_pop stays unchanged until next training
            }])
            
            logger.info(f"Updated Meilisearch ratings for book {book_id}: {num_ratings} ratings, avg {avg_rating}")
            
        except Exception as e:
            logger.error(f"Meilisearch update failed for book {book_id}: {e}")
        finally:
            db.close()
    
    # Run in background thread
    Thread(target=_update).start
