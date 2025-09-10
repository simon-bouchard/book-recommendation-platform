# app/agents/internal_tools.py
import json
from typing import Optional
from sqlalchemy.orm import Session
from models.recommender_strategy import WarmRecommender  # uses RecommendationEngine + ALSCandidateGenerator

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
