# app/agents/internal_tools.py
import json
from typing import Optional
from sqlalchemy.orm import Session
from types import SimpleNamespace

from models.recommender_strategy import WarmRecommender  # uses RecommendationEngine + ALSCandidateGenerator
from models.recommender_strategy import ColdRecommender
from models.shared_utils import get_read_books

def make_als_pool_tool(current_user, db: Session):
    """
    Returns a callable for ToolRegistry.
    Input format: 'top_k' (string, optional). Uses the *current_user* and *db* from closure.
    Output: JSON array string of your existing warm rec dicts.
    """
    def _tool(arg: str = "") -> str:
        try:
            k = int(arg.strip()) if arg and arg.strip().isdigit() else 100  # large pool for LLM curation
            recs = WarmRecommender().recommend(user=current_user, db=db, top_k=k)
            return json.dumps(recs, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"als_recs failed: {e}"})
    return _tool

# app/agents/internal_tools.py

import json
from typing import Optional
from sqlalchemy.orm import Session
from models.recommender_strategy import WarmRecommender  # already there

def make_return_book_ids_tool():
    """
    Finalization tool: accept item_idx list and echo back normalized IDs.
    Input can be a JSON list (preferred) or a comma-separated string.
    Output: JSON {"book_ids": [int, ...]}
    """
    def _tool(arg: str = "") -> str:
        try:
            arg = (arg or "").strip()
            if not arg:
                return json.dumps({"book_ids": []})

            # Try JSON first
            ids = None
            if arg.startswith("["):
                ids = json.loads(arg)
            else:
                ids = [x for x in arg.split(",") if x.strip()]

            norm = []
            for x in ids:
                # support numbers or strings
                norm.append(int(x))
            # de-dup while preserving order
            seen = set()
            deduped = []
            for i in norm:
                if i not in seen:
                    seen.add(i)
                    deduped.append(i)
            return json.dumps({"book_ids": deduped})
        except Exception as e:
            return json.dumps({"error": f"invalid ids: {e}"})
    return _tool

def make_subject_hybrid_pool_tool(current_user, db: Session):
    """
    Subject-hybrid candidate pool (mixed only) with adjustable top_k and w.
    Input: either empty string (use profile subjects) OR a JSON object:
        {"top_k": 200, "fav_subjects_idxs": [12, 77, 401], "w": 0.6}
    Output: JSON array of book dicts with
      {item_idx,title,author,year,cover_id,isbn,book_avg_rating,book_num_ratings,score}
    """
    def _tool(arg: str = "") -> str:
        try:
            # ---- Parse input ----
            top_k = 200
            subj_override: Optional[list[int]] = None
            w = 0.6  # default blend

            arg = (arg or "").strip()
            if arg:
                if arg.startswith("{"):
                    obj = json.loads(arg)
                    if "top_k" in obj and isinstance(obj["top_k"], int):
                        top_k = int(obj["top_k"])
                    if "fav_subjects_idxs" in obj and isinstance(obj["fav_subjects_idxs"], list):
                        subj_override = [int(x) for x in obj["fav_subjects_idxs"] if str(x).strip() != ""]
                    if "w" in obj:
                        try:
                            w_val = float(obj["w"])
                            # clamp to [0,1]
                            if w_val < 0: w_val = 0.0
                            if w_val > 1: w_val = 1.0
                            w = w_val
                        except Exception:
                            pass
                elif arg.isdigit():
                    top_k = int(arg)

            # ---- Build a safe user object for the engine ----
            if current_user is None:
                user_obj = SimpleNamespace(user_id=None, fav_subjects_idxs=[])
            elif subj_override is not None:
                user_obj = SimpleNamespace(
                    user_id=getattr(current_user, "user_id", None),
                    fav_subjects_idxs=subj_override
                )
            else:
                user_obj = current_user

            # ---- Generate pool via existing cold pipeline (mixed only) ----
            recs = ColdRecommender().recommend(
                user=user_obj,
                db=db,
                top_k=top_k,
                top_k_bayes=0,
                top_k_sim=0,
                top_k_mixed=max(top_k, 200),
                w=w,
            )

            # Optional: drop items the user already interacted with, if we know a user_id
            real_user_id = getattr(current_user, "user_id", None)
            if real_user_id is not None:
                already = get_read_books(user_id=real_user_id, db=db)
                recs = [r for r in recs if int(r.get("item_idx", -1)) not in already]

            # Trim to top_k
            recs = recs[:top_k]
            return json.dumps(recs, ensure_ascii=False)

        except Exception as e:
            return json.dumps({"error": f"subject_hybrid_pool failed: {e}"})

    return _tool
