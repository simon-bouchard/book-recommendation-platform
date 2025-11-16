# app/agents/tools/recsys/native_tools.py
"""
Modernized internal recommendation tools with output standardization.
All retrieval tools return consistent schema with enrichment data where available.
"""
from typing import Optional
from sqlalchemy.orm import Session
from types import SimpleNamespace

from models.recommender_strategy import WarmRecommender, ColdRecommender
from models.shared_utils import get_read_books, ModelStore
from app.semantic_index.search import SemanticSearcher
from app.agents.settings import settings

from ..native_tool import tool, ToolCategory, ToolDefinition


class InternalTools:
    """Factory for internal recommendation tools with standardized outputs."""
    
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
        self._tone_map: Optional[dict[int, str]] = None
    
    def _get_semantic_searcher(self) -> SemanticSearcher:
        """Lazy-load semantic searcher."""
        if self._semantic_searcher is None:
            self._semantic_searcher = SemanticSearcher(
                dir_path="models/data/enriched_v1",
                embedder=settings.embedder
            )
        return self._semantic_searcher
    
    def _ensure_tone_map(self) -> dict[int, str]:
        """
        Lazy-load tone ID to name mapping from database.
        
        Cached for lifetime of tool instance to avoid repeated queries.
        Single query returns ~36 rows (all tones in ontology).
        
        Returns:
            Dictionary mapping tone_id (int) to tone name (str)
        """
        if self._tone_map is None and self.db:
            from app.table_models import Tone
            
            try:
                tones = self.db.query(Tone.tone_id, Tone.name).all()
                self._tone_map = {tone_id: name for tone_id, name in tones}
            except Exception:
                # Fallback to empty map if query fails
                self._tone_map = {}
        
        return self._tone_map or {}
    
    def _standardize_tool_output(self, raw_results: list[dict]) -> list[dict]:
        """
        Standardize tool outputs to consistent schema.
        
        Ensures all retrieval tools return same field structure:
        - Core metadata: item_idx, title, author, year, num_ratings (always present)
        - Enrichment: subjects, tones, genre, vibe (optional, from semantic_search)
        - Internal: score (tool's relevance score, kept for debugging)
        
        Operations:
        1. Resolve tone_ids → tone names (from cached DB map)
        2. Add num_ratings from books.pkl (via ModelStore, in-memory)
        3. Standardize field names (vibe not description)
        4. Handle missing fields gracefully
        
        Args:
            raw_results: Raw output from recommender or searcher
            
        Returns:
            Standardized list of book dicts with consistent schema
        """
        if not raw_results:
            return []
        
        # Load shared resources once
        store = ModelStore()
        book_meta = store.get_book_meta()  # books.pkl, cached in-memory
        tone_map = self._ensure_tone_map()  # Cached DB query result
        
        standardized = []
        
        for book in raw_results:
            # Pass through error results unchanged
            if "error" in book:
                standardized.append(book)
                continue
            
            item_idx = book.get("item_idx")
            if not item_idx:
                continue
            
            # Get num_ratings from books.pkl (always available, zero queries)
            num_ratings = 0
            if item_idx in book_meta.index:
                num_ratings = int(book_meta.loc[item_idx].get("num_ratings", 0))
            
            # Resolve tone IDs to names (only if present - semantic_search has this)
            tone_ids = book.get("tone_ids", [])
            tone_names = []
            if tone_ids:
                tone_names = [
                    tone_map.get(tid) 
                    for tid in tone_ids 
                    if tid in tone_map
                ]
            
            # Build standardized dict
            standardized.append({
                # Core metadata (always present)
                "item_idx": item_idx,
                "title": book.get("title", ""),
                "author": book.get("author", ""),
                "year": book.get("year"),
                "num_ratings": num_ratings,
                
                # Enrichment metadata (optional - only semantic_search provides)
                "subjects": book.get("subjects", []),
                "tones": tone_names,
                "genre": book.get("genre"),
                "vibe": book.get("vibe"),
                
                # Internal only (not sent to curation)
                "score": book.get("score", 0.0),
            })
        
        return standardized
    
    def get_context_tools(self) -> list[ToolDefinition]:
        """
        Get context tools for PlannerAgent.
        
        These tools provide user preference data to inform strategy decisions.
        They do NOT retrieve books.
        
        Returns:
            List of context tools (empty if profile not allowed)
        """
        if not self.allow_profile or not self.current_user or not self.db:
            return []
        
        return self._create_user_context_tools()
    
    def get_retrieval_tools(self, is_warm: bool) -> list[ToolDefinition]:
        """
        Get retrieval tools for CandidateGeneratorAgent.
        
        These tools retrieve candidate books based on different strategies.
        All tools return standardized output format.
        
        Args:
            is_warm: Whether user has enough ratings for collaborative filtering
            
        Returns:
            List of retrieval tools
        """
        tools = []
        
        # Always available retrieval tools
        tools.append(self._create_semantic_search_tool())
        tools.append(self._create_subject_id_search_tool())
        tools.append(self._create_popular_books_tool())
        
        # Conditional on user state
        if is_warm and self.current_user and self.db:
            tools.append(self._create_als_recs_tool())
        
        if self.current_user and self.db:
            tools.append(self._create_subject_hybrid_tool())
        
        return tools
    
    def get_tools(self, is_warm: bool) -> list[ToolDefinition]:
        """
        Get all available internal tools based on user state.
        
        DEPRECATED: Use get_context_tools() or get_retrieval_tools() instead.
        This method is kept for backward compatibility.
        
        Args:
            is_warm: Whether user has enough ratings for warm recommendations
            
        Returns:
            List of all available tools
        """
        tools = []
        
        # Add context tools
        tools.extend(self.get_context_tools())
        
        # Add retrieval tools
        tools.extend(self.get_retrieval_tools(is_warm))
        
        # Add the finalizer tool
        tools.append(self._create_return_book_ids_tool())
        
        return tools
    
    def _create_semantic_search_tool(self) -> ToolDefinition:
        """Semantic search over book embeddings with full enrichment metadata."""
        
        def semantic_search(query: str, top_k: int = 200) -> list[dict]:
            """
            Search books using semantic similarity.
            
            Returns books with full enrichment metadata from semantic_meta.json:
            - subjects, tones, genre, vibe (from enrichment pipeline)
            - Standardized to include num_ratings from books.pkl
            
            Args:
                query: Search query describing books to find
                top_k: Number of results (max 200)
                
            Returns:
                Standardized list of books with enrichment metadata
            """
            searcher = self._get_semantic_searcher()
            top_k = max(1, min(200, top_k))
            
            try:
                raw_results = searcher.search(query, top_k=top_k)
                
                # Build intermediate format with metadata from JSON
                intermediate = []
                for r in raw_results:
                    meta = r.get('meta', {})
                    intermediate.append({
                        'item_idx': r.get('book_id'),
                        'score': r.get('score'),
                        'title': meta.get('title'),
                        'author': meta.get('author'),
                        'subjects': meta.get('subjects', []),
                        'tone_ids': meta.get('tone_ids', []),  # Will be resolved to names
                        'vibe': meta.get('vibe'),
                        'genre': meta.get('genre'),
                    })
                
                # Standardize (adds num_ratings, resolves tones)
                return self._standardize_tool_output(intermediate)
                
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
            
            Returns basic metadata (title, author, year, num_ratings).
            No enrichment data since ALS is based on rating patterns, not content.
            
            Args:
                top_k: Number of candidates to generate
                
            Returns:
                Standardized list of books with basic metadata
            """
            if not self.current_user or not self.db:
                return [{"error": "ALS requires authenticated user with database"}]
            
            top_k = max(1, min(500, top_k))
            
            try:
                recommender = WarmRecommender()
                raw_results = recommender.recommend(
                    user=self.current_user,
                    db=self.db,
                    top_k=top_k
                )
                
                # Standardize (adds num_ratings, no enrichment data expected)
                return self._standardize_tool_output(raw_results)
                
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
            Returns basic metadata (title, author, year, num_ratings).
            
            Args:
                top_k: Number of candidates to generate
                fav_subjects_idxs: Subject indices to use (uses profile if None)
                weight: Blend weight for subject vs popularity (0.0-1.0)
                
            Returns:
                Standardized list of books excluding already-read items
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
                raw_results = recommender.recommend(
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
                    raw_results = [
                        r for r in raw_results
                        if int(r.get("item_idx", -1)) not in already_read
                    ]
                
                raw_results = raw_results[:top_k]
                
                # Standardize (adds num_ratings)
                return self._standardize_tool_output(raw_results)
                
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
            Returns different format than other tools (subject matches, not books).
            
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
    
    def _create_popular_books_tool(self) -> ToolDefinition:
        """Get popular books ranked by Bayesian average rating."""
        
        def popular_books(top_k: int = 100) -> list[dict]:
            """
            Get popular books ranked by Bayesian average rating.
            
            Use for cold users with vague queries when no other context available.
            Returns books sorted by rating quality + popularity.
            Returns basic metadata (title, author, year, num_ratings).
            
            Args:
                top_k: Number of popular books to return
                
            Returns:
                Standardized list of popular books with basic metadata
            """
            if not self.current_user or not self.db:
                return [{"error": "Popular books requires authenticated user"}]
            
            top_k = max(1, min(500, top_k))
            
            try:
                recommender = ColdRecommender()
                raw_results = recommender.recommend(
                    user=self.current_user,
                    db=self.db,
                    top_k=top_k,
                    top_k_bayes=max(top_k, 200),  # Pure Bayesian ranking
                    top_k_sim=0,
                    top_k_mixed=0,
                    w=0.0,  # No mixing
                )
                
                # Exclude already-read books
                user_id = getattr(self.current_user, "user_id", None)
                if user_id:
                    already_read = get_read_books(user_id=user_id, db=self.db)
                    raw_results = [
                        r for r in raw_results
                        if int(r.get("item_idx", -1)) not in already_read
                    ]
                
                raw_results = raw_results[:top_k]
                
                # Standardize (adds num_ratings)
                return self._standardize_tool_output(raw_results)
                
            except Exception as e:
                return [{"error": f"Popular books failed: {e}"}]
        
        return tool(
            name="popular_books",
            description="Get popular books ranked by Bayesian average rating for cold users",
            category=ToolCategory.INTERNAL,
            requires_auth=True,
            requires_db=True,
        )(popular_books)
    
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
