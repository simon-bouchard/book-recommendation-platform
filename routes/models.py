# routes/models.py
"""
API endpoints for ML model services (recommendations and similarity).
Provides clean interface to recommendation and similarity services.
Maintains backward compatibility with old API response formats.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session, joinedload
import logging
import time

from metrics import RECSYS_REQUESTS, RECSYS_LATENCY, SIMILARITY_REQUESTS, SIMILARITY_LATENCY

from app.database import get_db
from app.table_models import User as ORMUser

from models.services.recommendation_service import RecommendationService
from models.services.similarity_service import SimilarityService
from models.domain.user import User
from models.domain.config import RecommendationConfig, HybridConfig, RecommendationMode
from models.core.constants import PAD_IDX
from models.cache import cached_recommendations, cached_similarity
from models.client.registry import get_similarity_client

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# Recommendation Endpoint (backward compatible with old API)
# ============================================================================


@cached_recommendations
async def _compute_recommendations(
    user: str,
    _id: bool,
    top_n: int,
    mode: str,
    w: float,
    db: Session,
) -> list[dict]:
    """
    Compute personalized book recommendations.

    Pure computation layer, separated from the route handler so that telemetry
    in the handler fires on every request regardless of cache state.
    """
    user_query = db.query(ORMUser).options(joinedload(ORMUser.favorite_subjects))

    if _id:
        user_obj = user_query.filter(ORMUser.user_id == int(user)).first()
    else:
        user_obj = user_query.filter(ORMUser.username == user).first()

    if not user_obj:
        raise HTTPException(status_code=404, detail="User not found")

    fav_subjects = [s.subject_idx for s in user_obj.favorite_subjects]
    if not fav_subjects:
        fav_subjects = [PAD_IDX]

    domain_user = User(
        user_id=user_obj.user_id,
        fav_subjects=fav_subjects,
        country=user_obj.country,
        age=user_obj.age,
        filled_age=user_obj.filled_age,
    )

    config = RecommendationConfig(
        k=top_n,
        mode=mode,
        hybrid_config=HybridConfig(subject_weight=w),
    )

    service = RecommendationService()
    recommendations = await service.recommend(domain_user, config, db)

    return [
        {
            "item_idx": rec.item_idx,
            "title": rec.title,
            "score": rec.score,
            "book_avg_rating": rec.avg_rating,
            "book_num_ratings": rec.num_ratings,
            "cover_id": rec.cover_id,
            "author": rec.author,
            "year": rec.year,
            "isbn": rec.isbn,
        }
        for rec in recommendations
    ]


@router.get("/profile/recommend")
async def recommend_for_user(
    user: str = Query(..., description="User ID or username"),
    _id: bool = Query(True, description="If true, user param is user_id; if false, username"),
    top_n: int = Query(200, ge=1, le=500, description="Number of recommendations"),
    mode: str = Query(
        "auto", regex="^(auto|subject|behavioral)$", description="Recommendation mode"
    ),
    w: float = Query(0.6, ge=0, le=1, description="Subject weight for hybrid mode"),
    db: Session = Depends(get_db),
) -> list[dict]:
    """
    Generate personalized book recommendations.

    Returns list of books with format matching old API for backward compatibility.

    Modes:
    - auto: Select strategy based on user status (warm -> ALS, cold -> hybrid)
    - subject: Force subject-based recommendations
    - behavioral: Force ALS-based recommendations
    """
    start_time = time.time()
    try:
        result = await _compute_recommendations(user, _id, top_n, mode, w, db)
        RECSYS_REQUESTS.labels(mode=mode).inc()
        RECSYS_LATENCY.labels(mode=mode).observe(time.time() - start_time)
        return result
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error in recommend_for_user: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error in recommend_for_user: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# Similarity Endpoint (backward compatible with old API)
# ============================================================================


@cached_similarity
async def _compute_similar_books(
    item_idx: int,
    mode: str,
    alpha: float,
    top_k: int,
) -> list[dict]:
    """
    Compute similar books for a given item.

    Pure computation layer, separated from the route handler so that telemetry
    in the handler fires on every request regardless of cache state.
    """
    service = SimilarityService()
    return await service.get_similar(
        item_idx=item_idx,
        mode=mode,
        k=top_k,
        alpha=alpha,
    )


@router.get("/book/{item_idx}/similar")
async def get_similar_books(
    item_idx: int,
    mode: str = Query("subject", regex="^(subject|als|hybrid)$"),
    alpha: float = Query(0.6, ge=0, le=1),
    top_k: int = Query(200, ge=1, le=500),
) -> list[dict]:
    """
    Find similar books using subject, collaborative filtering, or hybrid similarity.

    Returns list matching old API format for backward compatibility.

    Modes:
    - subject: Semantic similarity (no filtering)
    - als: Collaborative filtering (10+ ratings filter)
    - hybrid: Blended subject + ALS (5+ ratings filter)

    Alpha (hybrid only): 0.0=pure subject, 1.0=pure ALS, 0.6=default
    """
    if mode in ("als", "hybrid"):
        resp = await get_similarity_client().has_book_als(item_idx)
        if not resp.has_als:
            raise HTTPException(
                status_code=422,
                detail="This book doesn't have als factors yet because of a lack of interactions/ratings. Als and hybrid similarity isn't available for this book yet.",
            )

    start_time = time.time()
    try:
        results = await _compute_similar_books(item_idx, mode, alpha, top_k)
        SIMILARITY_REQUESTS.labels(mode=mode).inc()
        SIMILARITY_LATENCY.labels(mode=mode).observe(time.time() - start_time)
        return results
    except ValueError as e:
        logger.error(f"Validation error in get_similar_books: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error in get_similar_books: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
