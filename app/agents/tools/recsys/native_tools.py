# app/agents/tools/recsys/native_tools.py
"""
Modernized internal recommendation tools using refactored models API.
All retrieval tools return consistent schema with enrichment data where available.
"""

from typing import Optional
from pandas.core.common import standardize_mapping
from sqlalchemy.orm import Session

from models.services.recommendation_service import RecommendationService
from models.domain.user import User
from models.domain.config import RecommendationConfig, HybridConfig
from models.data.queries import get_read_books
from models.data.loaders import load_book_meta, load_bayesian_scores, load_book_subject_embeddings
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
                dir_path="models/artifacts/semantic_indexes/enriched_v2",
                embedder=settings.embedder,
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
        1. Resolve tone_ids -> tone names (from cached DB map)
        2. Add num_ratings from book metadata (via loaders, in-memory)
        3. Standardize field names (vibe not description)
        4. Exclude empty enrichment fields to reduce token usage

        Args:
            raw_results: Raw output from service or searcher

        Returns:
            Standardized list of book dicts with consistent schema
        """
        if not raw_results:
            return []

        # Load shared resources once
        book_meta = load_book_meta(use_cache=True)

        standardized = []

        for book in raw_results:
            # Pass through error results unchanged
            if "error" in book:
                standardized.append(book)
                continue

            item_idx = book.get("item_idx")
            if not item_idx:
                continue

            # Get num_ratings from book metadata (always available, zero queries)
            num_ratings = 0
            if item_idx in book_meta.index:
                num_ratings = int(book_meta.loc[item_idx].get("book_num_ratings", 0))

            # Build standardized dict - only include fields with content
            result = {
                # Core metadata (always present)
                "item_idx": item_idx,
                "title": book.get("title", ""),
                "author": book.get("author", ""),
                "year": book.get("year"),
                "num_ratings": num_ratings,
                # Internal only (not sent to curation)
                "score": book.get("score", 0.0),
            }

            standardized.append(result)

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

        # Conditional on database availability
        if self.db:
            tools.append(self._create_subject_hybrid_tool())

        # Conditional on user state (warm + authenticated)
        if is_warm and self.current_user and self.db:
            tools.append(self._create_als_recs_tool())

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
        """Semantic search with enriched metadata."""

        def semantic_search(
            query: str, top_k: int = 100, filters: Optional[dict] = None
        ) -> list[dict]:
            """
            Search books using semantic embeddings with optional subject filters.

            Use for specific queries about themes, topics, or vibes.
            Returns enriched metadata: subjects, tones, genre, vibe.
            Works for both authenticated and anonymous users.

            Args:
                query: Natural language search query
                top_k: Number of results to return
                filters: Optional dict with 'subjects' list for filtering

            Returns:
                Standardized list of books with enrichment metadata
            """
            top_k = max(1, min(500, top_k))

            try:
                searcher = self._get_semantic_searcher()

                """
                # Extract subject filters if provided
                subject_filter = None
                if filters and "subjects" in filters:
                    subject_filter = filters["subjects"]
                """

                results = searcher.search(query=query, top_k=top_k)

                for book in results:
                    if "meta" in book:
                        meta = book.pop("meta")
                        book.update(meta)

                # Results already have enrichment data from semantic index
                return self._standardize_tool_output(results)

            except Exception as e:
                return [{"error": f"Semantic search failed: {e}"}]

        return tool(
            name="book_semantic_search",
            description="Search books using semantic embeddings with enriched metadata",
            category=ToolCategory.INTERNAL,
        )(semantic_search)

    def _create_als_recs_tool(self) -> ToolDefinition:
        """ALS-based collaborative filtering recommendations."""

        def als_recs(top_k: int = 100) -> list[dict]:
            """
            Get collaborative filtering recommendations based on user's rating history.

            Use ONLY for warm users (10+ ratings).
            Returns books similar to what user has rated highly.
            Returns basic metadata (title, author, year, num_ratings).

            Args:
                top_k: Number of recommendations to return

            Returns:
                Standardized list of recommended books with basic metadata
            """
            if not self.current_user or not self.db:
                return [{"error": "ALS recommendations require authentication"}]

            top_k = max(1, min(500, top_k))

            try:
                # Convert ORM user to domain User
                domain_user = self._to_domain_user(self.current_user)

                # Use new recommendation service with behavioral mode
                service = RecommendationService()
                config = RecommendationConfig(k=top_k, mode="behavioral")

                recommendations = service.recommend(domain_user, config, self.db)

                # Convert RecommendedBook objects to dicts
                raw_results = [
                    {
                        "item_idx": rec.item_idx,
                        "title": rec.title,
                        "author": rec.author,
                        "year": rec.year,
                        "score": rec.score,
                        "num_ratings": rec.num_ratings,
                    }
                    for rec in recommendations
                ]

                return self._standardize_tool_output(raw_results)

            except Exception as e:
                return [{"error": f"ALS recommendations failed: {e}"}]

        return tool(
            name="als_recs",
            description="Get collaborative filtering recommendations for warm users",
            category=ToolCategory.INTERNAL,
            requires_auth=True,
            requires_db=True,
        )(als_recs)

    def _create_subject_hybrid_tool(self) -> ToolDefinition:
        """Subject-based recommendations with popularity blending."""

        def subject_hybrid(
            fav_subjects_idxs: list[str],
            top_k: int = 100,
            subject_weight: float = 0.6,
        ) -> list[dict]:
            """
            Get recommendations based on subject preferences with popularity blending.

            Use for cold users with known subject preferences.
            Blends subject similarity with popularity for quality results.
            Returns basic metadata (title, author, year, num_ratings).

            Args:
                subjects: List of subject names user is interested in
                top_k: Number of recommendations to return
                subject_weight: Weight for subject similarity (0-1), rest is popularity

            Returns:
                Standardized list of recommended books with basic metadata
            """
            if not self.db:
                return [{"error": "Subject hybrid requires database connection"}]

            if not fav_subjects_idxs:
                return [{"error": "Subject hybrid requires at least one subject"}]

            top_k = max(1, min(500, top_k))
            subject_weight = max(0.0, min(1.0, subject_weight))

            try:
                subject_indices = fav_subjects_idxs

                # Create domain user with subject preferences
                domain_user = self._create_user_with_subjects(subject_indices)

                # Use new recommendation service with subject mode
                service = RecommendationService()
                config = RecommendationConfig(
                    k=top_k,
                    mode="subject",
                    hybrid_config=HybridConfig(subject_weight=subject_weight),
                )

                recommendations = service.recommend(domain_user, config, self.db)

                # Convert RecommendedBook objects to dicts
                raw_results = [
                    {
                        "item_idx": rec.item_idx,
                        "title": rec.title,
                        "author": rec.author,
                        "year": rec.year,
                        "score": rec.score,
                        "num_ratings": rec.num_ratings,
                    }
                    for rec in recommendations
                ]

                return self._standardize_tool_output(raw_results)

            except Exception as e:
                return [{"error": f"Subject hybrid failed: {e}"}]

        return tool(
            name="subject_hybrid_pool",
            description="Get subject-based recommendations with popularity blending",
            category=ToolCategory.INTERNAL,
            requires_db=True,
        )(subject_hybrid)

    def _create_subject_id_search_tool(self) -> ToolDefinition:
        """Subject ID search using 3-gram TF-IDF."""
        from .subject_search import make_subject_id_search_tool

        def subject_id_search(phrases: list[str], top_k: int = 5) -> list[dict]:
            """
            Resolve free-text subject phrases to database subject IDs.

            Use to map user's subject interests to database subjects.
            Returns subject_idx, subject name, and match score.

            Args:
                phrases: List of subject phrases to resolve
                top_k: Number of matches per phrase

            Returns:
                List of dicts with phrase and candidates
            """
            if not self.db:
                return [{"error": "Subject ID search requires database"}]

            top_k = max(1, min(10, top_k))

            # Use the existing implementation
            tool_func = make_subject_id_search_tool(self.db)

            # Call with JSON input
            import json

            input_json = json.dumps({"phrases": phrases, "top_k": top_k})
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
            Works for both authenticated and anonymous users.
            Returns basic metadata (title, author, year, num_ratings).

            Args:
                top_k: Number of popular books to return

            Returns:
                Standardized list of popular books with basic metadata
            """
            if not self.db:
                return [{"error": "Popular books requires database connection"}]

            top_k = max(1, min(500, top_k))

            try:
                # Create user for popularity fallback
                # For anonymous: use None values
                # For authenticated: use real user
                if self.current_user:
                    domain_user = self._to_domain_user(self.current_user)
                else:
                    # Anonymous user - popularity generator doesn't need user data
                    from models.core.constants import PAD_IDX

                    domain_user = User(user_id=-1, fav_subjects=[PAD_IDX])

                # Use new recommendation service
                # The service will automatically use popularity generator
                # when user has no preferences (which is true for anonymous users)
                service = RecommendationService()
                config = RecommendationConfig(k=top_k, mode="auto")

                recommendations = service.recommend(domain_user, config, self.db)

                # Convert RecommendedBook objects to dicts
                raw_results = [
                    {
                        "item_idx": rec.item_idx,
                        "title": rec.title,
                        "author": rec.author,
                        "year": rec.year,
                        "score": rec.score,
                        "num_ratings": rec.num_ratings,
                    }
                    for rec in recommendations
                ]

                return self._standardize_tool_output(raw_results)

            except Exception as e:
                return [{"error": f"Popular books failed: {e}"}]

        return tool(
            name="popular_books",
            description="Get popular books ranked by Bayesian average rating for cold users",
            category=ToolCategory.INTERNAL,
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

        tools.append(
            tool(
                name="user_profile",
                description="Get user's favorite subjects (requires consent)",
                category=ToolCategory.INTERNAL,
                requires_auth=True,
                requires_db=True,
            )(user_profile)
        )

        # Recent interactions tool
        def recent_interactions(limit: int = 5, include_comments: bool = False) -> dict:
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
                    items.append(
                        {
                            "title": interaction.get("title"),
                            "rating": interaction.get("rating"),
                            "date": interaction.get("date"),
                            "comment": interaction.get("comment", "") if include_comments else "",
                        }
                    )
                return {"interactions": items}
            except Exception as e:
                return {"error": f"Interactions fetch failed: {e}"}

        tools.append(
            tool(
                name="recent_interactions",
                description="Get user's recent rated books (requires consent)",
                category=ToolCategory.INTERNAL,
                requires_auth=True,
                requires_db=True,
            )(recent_interactions)
        )

        return tools

    def _to_domain_user(self, orm_user) -> User:
        """
        Convert ORM user to domain User object.

        Args:
            orm_user: SQLAlchemy User object

        Returns:
            Domain User object with favorite subjects
        """
        from models.core.constants import PAD_IDX

        # Get favorite subjects
        fav_subjects = []
        if hasattr(orm_user, "favorite_subjects"):
            fav_subjects = [s.subject_idx for s in orm_user.favorite_subjects]

        if not fav_subjects:
            fav_subjects = [PAD_IDX]

        return User(
            user_id=orm_user.user_id,
            fav_subjects=fav_subjects,
            country=getattr(orm_user, "country", None),
            age=getattr(orm_user, "age", None),
            filled_age=getattr(orm_user, "filled_age", None),
        )

    def _create_user_with_subjects(self, subject_indices: list[int]) -> User:
        """
        Create a domain User with given subject preferences.

        Used for subject_hybrid when we need to create a user with specific subjects.

        Args:
            subject_indices: List of subject indices

        Returns:
            Domain User object
        """
        # Use a temporary user ID (negative to avoid collision)
        # The service layer doesn't need a real user_id for cold recommendations
        user_id = -1

        # If we have a real user, use their ID
        if self.current_user:
            user_id = getattr(self.current_user, "user_id", -1)

        return User(
            user_id=user_id, fav_subjects=subject_indices, country=None, age=None, filled_age=None
        )
