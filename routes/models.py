# routes/models.py
"""
API endpoints for ML model services (recommendations and similarity).
Provides clean interface to recommendation and similarity services.
Maintains backward compatibility with old API response formats.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session, joinedload
import os
import logging, time

from metrics import RECSYS_REQUESTS, RECSYS_LATENCY, SIMILARITY_REQUESTS, SIMILARITY_LATENCY

from app.database import get_db
from app.table_models import User as ORMUser

from models.services.recommendation_service import RecommendationService
from models.services.similarity_service import SimilarityService
from models.domain.user import User
from models.domain.config import RecommendationConfig, HybridConfig, RecommendationMode
from models.core.constants import PAD_IDX
from models.cache import cached_recommendations, cached_similarity

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# Recommendation Endpoint (backward compatible with old API)
# ============================================================================


@router.get("/profile/recommend")
@cached_recommendations
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
        # Fetch user from database
        user_query = db.query(ORMUser).options(joinedload(ORMUser.favorite_subjects))

        if _id:
            user_obj = user_query.filter(ORMUser.user_id == int(user)).first()
        else:
            user_obj = user_query.filter(ORMUser.username == user).first()

        if not user_obj:
            raise HTTPException(status_code=404, detail="User not found")

        # Convert ORM user to domain User
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

        # Build configuration (cast mode string to Literal type)
        config = RecommendationConfig(
            k=top_n,
            mode=mode,  
            hybrid_config=HybridConfig(subject_weight=w),
        )

        # Generate recommendations
        service = RecommendationService()
        recommendations = service.recommend(domain_user, config, db)

        RECSYS_REQUESTS.labels(mode=mode).inc()
        RECSYS_LATENCY.labels(mode=mode).observe(time.time() - start_time)

        # Convert to OLD API format (flat list with old field names)
        return [
            {
                "item_idx": rec.item_idx,
                "title": rec.title,
                "score": rec.score,
                "book_avg_rating": rec.avg_rating,  # Old field name
                "book_num_ratings": rec.num_ratings,  # Old field name
                "cover_id": rec.cover_id,
                "author": rec.author,
                "year": rec.year,
                "isbn": rec.isbn,
            }
            for rec in recommendations
        ]

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


@router.get("/book/{item_idx}/similar")
@cached_similarity
def get_similar_books(
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
    start_time = time.time()

    # Check if book has required data for mode
    if mode == "als":
        from models.data.loaders import load_als_factors

        _, _, _, book_row_map = load_als_factors(use_cache=True)
        book_ids = set(book_row_map.values())

        if item_idx not in book_ids:
            raise HTTPException(
                status_code=422,
                detail="Behavioral similarity unavailable for this book (no ALS data). Try Subject mode.",
            )

    if mode == "hybrid":
        # Hybrid needs both subject and ALS, but can fall back if ALS missing
        pass

    try:
        service = SimilarityService()

        results = service.get_similar(
            item_idx=item_idx,
            mode=mode,
            k=top_k,
            alpha=alpha,
            min_rating_count=None,  
            filter_candidates=True,
        )

        SIMILARITY_REQUESTS.labels(mode=mode).inc()
        SIMILARITY_LATENCY.labels(mode=mode).observe(time.time() - start_time)

        # Return in OLD API format (flat list of dicts)
        return results  

    except ValueError as e:
        logger.error(f"Validation error in get_similar_books: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Error in get_similar_books: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# Admin Endpoint
# ============================================================================


@router.post("/admin/reload_models")
def reload_models_endpoint(secret: str = Query(...)):
    """
    Reload all model artifacts from disk.

    Clears all in-memory caches and reloads artifacts for model updates.
    Requires ADMIN_SECRET environment variable for authorization.
    """
    if secret != os.getenv("ADMIN_SECRET"):
        raise HTTPException(status_code=403, detail="Forbidden")

    try:
        from models.data.loaders import clear_cache, preload_all_artifacts
        from models.infrastructure.subject_embedder import SubjectEmbedder
        from models.infrastructure.als_model import ALSModel
        from models.infrastructure.similarity_indices import reset_indices

        # Clear all caches
        clear_cache()
        SubjectEmbedder.reset()
        ALSModel.reset()
        reset_indices()  # NEW: Clear FAISS indices

        # Preload artifacts (now includes indices)
        preload_all_artifacts()

        logger.info("Models reloaded successfully")
        return {"status": "reloaded"}

    except Exception as e:
        logger.error(f"Reload failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")
