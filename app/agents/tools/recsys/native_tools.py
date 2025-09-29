# app/agents/tools/recsys/native_tools.py
"""
Modernized internal recommendation tools with clean interfaces.
No string parsing gymnastics - just typed function signatures.
"""
from typing import Optional
from sqlalchemy.orm import Session
from types import SimpleNamespace

from models.recommender_strategy import WarmRecommender, ColdRecommender
from models.shared_utils import get_read_books
from app.semantic_index.search import SemanticSearcher
from app.agents.settings import settings

from ..native_tool import tool, ToolCategory, ToolDefinition


class InternalTools:
    """Factory for internal recommendation tools."""
    
    def __init__(
        self,
        current_user=None,
        db: Optional[Session] = None,
        user_num_ratings: int = 0,
        allow_profile: bool = False,
    ):
        self.current_user = current_user
        self.db = db
        self.user_num_ratings = user_num_ratings
        self.allow_profile = allow_profile
        self._semantic_searcher: Optional[SemanticSearcher] = None
    
    def _get_semantic_searcher(self) -> SemanticSearcher:
        """Lazy-load semantic searcher."""
        if self._semantic_searcher is None:
            self._semantic_searcher = SemanticSearcher(
                dir_path="models/data",
                embedder=settings.embedder
            )
        return self._semantic_searcher
    
    def get_tools(self, is_warm: bool) -> list[ToolDefinition]:
        """
        Get available internal tools based on user state.
        
        Args:
            is_warm: Whether user has enough ratings for warm recommendations
            
        Returns:
            List of available tools
        """
        tools = []
        
        # Always available
        tools.append(self._create_semantic_search_tool())
        tools.append(self._create_return_book_ids_tool())
        
        # Conditional on user state
        if is_warm and self.current_user and self.db:
            tools.append(self._create_als_recs_tool())
        
        if self.current_user and self.db:
            tools.append(self._create_subject_hybrid_tool())
            tools.append(self._create_subject_id_search_tool())
        
        if self.allow_profile and self.current_user and self.db:
            tools.extend(self._create_user_context_tools())
        
        return tools
    
    def _create_semantic_search_tool(self) -> ToolDefinition:
        """Semantic search over book embeddings."""
        
        def semantic_search(query: str, top_k: int = 200) -> list[dict]:
            """
            Search books using semantic similarity.
            
            Best for: Finding books by description, vibe, or themes.
            
            Args:
                query: Free-text description of desired books
                top_k: Number of results to return (max 200)
                
            Returns:
                List of book candidates with metadata and scores
            """
            searcher = self._get_semantic_searcher()
            top_k = max(1, min(200, top_k))
            
            try:
                results = searcher.search(query, top_k=top_k)
                # Rename book_id to item_idx for consistency
                for r in results:
                    r['item_idx'] = r.pop('book_id', None)
                return results
            except Exception as e:
                return [{"error": f"Semantic search failed: {e}"}]
        
        return tool(
            name="book_semantic_search",
            description="Semantic search for books by description, vibe, or themes",
            category=ToolCategory.INTERNAL,
        )(semantic_search)
    
    def _create_als_recs_tool(self) -> ToolDefinition:
        """ALS collaborative filtering for warm users."""
        
        def als_recommendations(top_k: int = 200) -> list[dict]:
            """
            Generate personalized recommendations using collaborative filtering.
            
            Requires: User must have rated at least 10 books.
            
            Args:
                top_k: Number of candidates to generate
                
            Returns:
                List of recommended books with scores and metadata
            """
            if not self.current_user or not self.db:
                return [{"error": "ALS requires authenticated user with database"}]
            
            top_k = max(1, min(500, top_k))
            
            try:
                recommender = WarmRecommender()
                results = recommender.recommend(
                    user=self.current_user,
                    db=self.db,
                    top_k=top_k
                )
                return results
            except Exception as e:
                return [{"error": f"ALS recommendations failed: {e}"}]
        
        return tool(
            name="als_recs",
            description="Personalized collaborative filtering recommendations for warm users",
            category=ToolCategory.INTERNAL,
            requires_auth=True,
            requires_db=True,
        )(als_recommendations)
    
    def _create_subject_hybrid_tool(self) -> ToolDefinition:
        """Subject-based hybrid recommendations."""
        
        def subject_hybrid_pool(
            top_k: int = 200,
            fav_subjects_idxs: Optional[list[int]] = None,
            weight: float = 0.6,
        ) -> list[dict]:
            """
            Generate recommendations based on subject preferences.
            
            Uses a hybrid approach combining subject matching and Bayesian popularity.
            
            Args:
                top_k: Number of candidates to generate
                fav_subjects_idxs: Subject indices to use (uses profile if None)
                weight: Blend weight for subject vs popularity (0.0-1.0)
                
            Returns:
                List of recommended books excluding already-read items
            """
            if not self.current_user or not self.db:
                return [{"error": "Subject hybrid requires authenticated user"}]
            
            top_k = max(1, min(500, top_k))
            weight = max(0.0, min(1.0, weight))
            
            # Build user object for recommender
            if fav_subjects_idxs is not None:
                user_obj = SimpleNamespace(
                    user_id=getattr(self.current_user, "user_id", None),
                    fav_subjects_idxs=fav_subjects_idxs
                )
            else:
                user_obj = self.current_user
            
            try:
                recommender = ColdRecommender()
                results = recommender.recommend(
                    user=user_obj,
                    db=self.db,
                    top_k=top_k,
                    top_k_bayes=0,
                    top_k_sim=0,
                    top_k_mixed=max(top_k, 200),
                    w=weight,
                )
                
                # Exclude already-read books
                user_id = getattr(self.current_user, "user_id", None)
                if user_id:
                    already_read = get_read_books(user_id=user_id, db=self.db)
                    results = [
                        r for r in results
                        if int(r.get("item_idx", -1)) not in already_read
                    ]
                
                return results[:top_k]
                
            except Exception as e:
                return [{"error": f"Subject hybrid failed: {e}"}]
        
        return tool(
            name="subject_hybrid_pool",
            description="Generate recommendations based on subject preferences with popularity blending",
            category=ToolCategory.INTERNAL,
            requires_auth=True,
            requires_db=True,
        )(subject_hybrid_pool)
    
    def _create_subject_id_search_tool(self) -> ToolDefinition:
        """Resolve subject phrases to IDs."""
        
        def subject_id_search(
            phrases: list[str],
            top_k: int = 5
        ) -> list[dict]:
            """
            Resolve free-text subject phrases to subject indices.
            
            Uses TF-IDF similarity to match phrases to database subjects.
            
            Args:
                phrases: List of subject phrases to resolve
                top_k: Number of candidates per phrase
                
            Returns:
                List of matches per phrase with scores
            """
            if not self.db:
                return [{"error": "Subject search requires database"}]
            
            from .subject_search import make_subject_id_search_tool
            
            # Use the existing implementation
            tool_func = make_subject_id_search_tool(self.db)
            
            # Call with JSON input
            import json
            input_json = json.dumps({
                "phrases": phrases,
                "top_k": max(1, min(10, top_k))
            })
            
            result_json = tool_func(input_json)
            return json.loads(result_json)
        
        return tool(
            name="subject_id_search",
            description="Resolve free-text subject phrases to database subject indices",
            category=ToolCategory.INTERNAL,
            requires_db=True,
        )(subject_id_search)
    
    def _create_return_book_ids_tool(self) -> ToolDefinition:
        """Finalize and return selected book IDs."""
        
        def return_book_ids(book_ids: list[int]) -> dict[str, list[int]]:
            """
            Finalize selected book recommendations.
            
            This tool should be called last to return the curated list
            of book IDs to show the user.
            
            Args:
                book_ids: List of item_idx values to return
                
            Returns:
                Dictionary with deduplicated book_ids list
            """
            # Deduplicate while preserving order
            seen = set()
            deduped = []
            for book_id in book_ids:
                if book_id not in seen:
                    seen.add(book_id)
                    deduped.append(int(book_id))
            
            return {"book_ids": deduped}
        
        return tool(
            name="return_book_ids",
            description="Finalize and return selected book recommendations to user",
            category=ToolCategory.INTERNAL,
        )(return_book_ids)
    
    def _create_user_context_tools(self) -> list[ToolDefinition]:
        """User profile and interaction history tools (requires consent)."""
        from app.agents.user_context import fetch_user_context
        
        tools = []
        
        # User profile tool
        def user_profile(limit: int = 5) -> dict:
            """
            Get user's favorite subjects.
            
            Requires: User consent for profile access.
            
            Args:
                limit: Maximum number of subjects to return (max 5)
                
            Returns:
                Dictionary with fav_subjects list
            """
            if not self.current_user or not self.db:
                return {"error": "User profile requires authentication"}
            
            user_id = getattr(self.current_user, "user_id", None)
            if not user_id:
                return {"fav_subjects": []}
            
            limit = max(1, min(5, limit))
            
            try:
                ctx = fetch_user_context(self.db, user_id, limit=limit)
                return {"fav_subjects": ctx.get("fav_subjects", [])[:limit]}
            except Exception as e:
                return {"error": f"Profile fetch failed: {e}"}
        
        tools.append(tool(
            name="user_profile",
            description="Get user's favorite subjects (requires consent)",
            category=ToolCategory.INTERNAL,
            requires_auth=True,
            requires_db=True,
        )(user_profile))
        
        # Recent interactions tool
        def recent_interactions(
            limit: int = 5,
            include_comments: bool = False
        ) -> dict:
            """
            Get user's recent book interactions.
            
            Requires: User consent for profile access.
            
            Args:
                limit: Maximum number of interactions (max 5)
                include_comments: Whether to include user comments
                
            Returns:
                Dictionary with interactions list
            """
            if not self.current_user or not self.db:
                return {"error": "Recent interactions requires authentication"}
            
            user_id = getattr(self.current_user, "user_id", None)
            if not user_id:
                return {"interactions": []}
            
            limit = max(1, min(5, limit))
            
            try:
                ctx = fetch_user_context(self.db, user_id, limit=limit)
                items = []
                for interaction in ctx.get("interactions", [])[:limit]:
                    items.append({
                        "title": interaction.get("title"),
                        "rating": interaction.get("rating"),
                        "date": interaction.get("date"),
                        "comment": interaction.get("comment", "") if include_comments else "",
                    })
                return {"interactions": items}
            except Exception as e:
                return {"error": f"Interactions fetch failed: {e}"}
        
        tools.append(tool(
            name="recent_interactions",
            description="Get user's recent rated books (requires consent)",
            category=ToolCategory.INTERNAL,
            requires_auth=True,
            requires_db=True,
        )(recent_interactions))
        
        return tools