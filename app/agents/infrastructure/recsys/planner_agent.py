# app/agents/infrastructure/recsys/planner_agent.py
"""
PlannerAgent for recommendation system query analysis and strategy planning.
First stage of the multi-agent pipeline: analyzes queries and determines retrieval strategy.

Simple single-call implementation - pre-fetches profile data and makes one LLM call.
"""

import json
from typing import Optional, Dict, Any

from langchain_core.messages import HumanMessage

from app.agents.domain.recsys_schemas import PlannerInput, PlannerStrategy
from app.agents.prompts.loader import read_prompt
from app.agents.settings import get_llm
from app.agents.logging import append_chatbot_log


class PlannerAgent:
    """
    Simple planning agent that makes a single LLM call to determine retrieval strategy.

    Pre-fetches user profile data if allowed, then asks LLM to analyze the query
    and recommend which retrieval tools to use. Returns a PlannerStrategy object
    for the CandidateGeneratorAgent to execute.

    No ReAct loop needed - this is pure query classification and tool selection.
    """

    def __init__(
        self,
        current_user=None,
        db=None,
        user_num_ratings: int = 0,
        has_als_recs_available: bool = False,
        allow_profile: bool = False,
    ):
        """
        Initialize PlannerAgent.

        Args:
            current_user: Current user object (for profile data fetching)
            db: Database session (for profile data fetching)
            user_num_ratings: Number of ratings user has (for strategy decisions)
            has_als_recs_available: Whether ALS collaborative filtering is available
            allow_profile: Whether agent can access user profile data
        """
        self.llm = get_llm(tier="medium", json_mode=True, temperature=0.0)
        self._ctx_user = current_user
        self._ctx_db = db
        self._user_num_ratings = user_num_ratings
        self._has_als_recs_available = has_als_recs_available
        self._allow_profile = allow_profile

    def execute(self, planner_input: PlannerInput) -> PlannerStrategy:
        """
        Analyze query and determine retrieval strategy.

        Process:
        1. Pre-fetch profile data if allowed (direct function call)
        2. Build prompt with all context (query, tools, profile, user stats)
        3. Single LLM call for strategy
        4. Parse JSON response into PlannerStrategy

        Args:
            planner_input: Query and available tools context

        Returns:
            PlannerStrategy with tool recommendations and reasoning

        Raises:
            ValueError: If JSON parsing fails or required fields missing
        """
        append_chatbot_log("\n=== PLANNER AGENT ===")
        append_chatbot_log(f"Query: {planner_input.query[:100]}...")
        append_chatbot_log(
            f"Context: ALS={self._has_als_recs_available}, "
            f"Profile={self._allow_profile}, Ratings={self._user_num_ratings}"
        )

        # Step 1: Pre-fetch profile data if allowed
        profile_data = None
        if self._allow_profile and self._ctx_user and self._ctx_db:
            profile_data = self._fetch_profile_data()
            if profile_data:
                append_chatbot_log(
                    f"Fetched profile: {len(profile_data.get('favorite_subjects', []))} subjects, "
                    f"{len(profile_data.get('recent_interactions', []))} interactions"
                )

        # Step 2: Build complete prompt with all context
        prompt = self._build_prompt(
            query=planner_input.query,
            available_tools=planner_input.available_retrieval_tools,
            profile_data=profile_data,
        )

        # Step 3: Single LLM call
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            response_text = response.content if hasattr(response, "content") else str(response)

            # Log raw response for debugging
            append_chatbot_log(f"[PLANNER RESPONSE] Raw text ({len(response_text)} chars)")
            if len(response_text) <= 500:
                append_chatbot_log(response_text)
            else:
                append_chatbot_log(f"{response_text[:500]}... (truncated)")

        except Exception as e:
            append_chatbot_log(f"[PLANNER ERROR] LLM call failed: {e}")
            raise RuntimeError(f"PlannerAgent LLM call failed: {e}") from e

        # Step 4: Parse JSON response
        try:
            strategy = self._parse_strategy_response(response_text, profile_data)

            append_chatbot_log(f"[PLANNER SUCCESS] Strategy: {strategy.reasoning[:150]}")
            append_chatbot_log(
                f"[PLANNER SUCCESS] Recommended: {', '.join(strategy.recommended_tools)}"
            )
            append_chatbot_log(f"[PLANNER SUCCESS] Fallback: {', '.join(strategy.fallback_tools)}")

            return strategy

        except json.JSONDecodeError as e:
            append_chatbot_log(f"[PLANNER ERROR] JSON parse failed: {e}")
            raise ValueError(f"Failed to parse planner strategy JSON: {e}") from e

        except KeyError as e:
            append_chatbot_log(f"[PLANNER ERROR] Missing required field: {e}")
            raise ValueError(f"Failed to parse planner strategy - missing field: {e}") from e

        except Exception as e:
            append_chatbot_log(f"[PLANNER ERROR] Unexpected parsing error: {e}")
            raise ValueError(f"Failed to parse planner strategy: {e}") from e

    def _fetch_profile_data(self) -> Optional[Dict[str, Any]]:
        """
        Directly fetch profile data from database.

        Calls the user_context module directly instead of going through tool wrappers.
        Returns None if user has no profile data or fetching fails.

        Returns:
            Dictionary with favorite_subjects and recent_interactions, or None
        """
        from app.agents.user_context import fetch_user_context

        user_id = getattr(self._ctx_user, "user_id", None)
        if not user_id:
            return None

        try:
            ctx = fetch_user_context(self._ctx_db, user_id, limit=5)

            # Extract relevant data
            favorite_subjects = ctx.get("fav_subjects", [])
            recent_interactions = ctx.get("interactions", [])[:3]  # Keep top 3

            # Only return if we have any data
            if not favorite_subjects and not recent_interactions:
                return None

            return {
                "favorite_subjects": favorite_subjects,
                "recent_interactions": recent_interactions,
            }

        except Exception as e:
            append_chatbot_log(f"[PLANNER WARNING] Profile fetch failed: {e}")
            return None

    def _build_prompt(
        self, query: str, available_tools: list[str], profile_data: Optional[Dict[str, Any]]
    ) -> str:
        """
        Build complete prompt with all context for strategy decision.

        Args:
            query: User's query
            available_tools: List of available retrieval tool names
            profile_data: User profile data if available

        Returns:
            Complete prompt string with system instructions and context
        """
        # Load base system prompt (decision logic)
        system_prompt = read_prompt("recsys.planner.md")

        # Build context sections
        context_parts = []

        # User context section
        context_parts.append("# USER CONTEXT")
        context_parts.append("")
        context_parts.append(f"**User has {self._user_num_ratings} ratings**")
        context_parts.append(
            f"**ALS collaborative filtering available:** {self._has_als_recs_available}"
        )
        context_parts.append("")

        # Profile data section (if available)
        if profile_data:
            context_parts.append("**User Profile Data:**")

            fav_subjects = profile_data.get("favorite_subjects", [])
            if fav_subjects:
                # Show both names and IDs clearly
                subject_display = []
                subject_ids = []

                for subj in fav_subjects[:5]:
                    if isinstance(subj, dict):
                        # New format: {"subject_idx": 12, "subject": "Science Fiction"}
                        subject_display.append(f"{subj['subject']} (ID: {subj['subject_idx']})")
                        subject_ids.append(subj["subject_idx"])
                    else:
                        # Old format: just strings (backward compatibility)
                        subject_display.append(str(subj))

                context_parts.append(f"- Favorite subjects: {', '.join(subject_display)}")

                # Provide IDs in easy-to-use format for tools
                if subject_ids:
                    context_parts.append(f"- Subject IDs for subject_hybrid_pool: {subject_ids}")

            interactions = profile_data.get("recent_interactions", [])
            if interactions:
                context_parts.append(f"- Recent interactions: {len(interactions)} books")
                for interaction in interactions:
                    title = interaction.get("title", "Unknown")[:40]
                    rating = interaction.get("rating", "?")
                    context_parts.append(f"  - {title} (rating: {rating})")

            context_parts.append("")

        # Available tools section
        context_parts.append("# AVAILABLE RETRIEVAL TOOLS")
        context_parts.append("")
        context_parts.append("The CandidateGeneratorAgent can use these tools:")
        for tool_name in available_tools:
            context_parts.append(f"- {tool_name}")
        context_parts.append("")

        # Query section
        context_parts.append("# USER QUERY")
        context_parts.append("")
        context_parts.append(query)
        context_parts.append("")

        # Instructions section
        context_parts.append("# YOUR TASK")
        context_parts.append("")
        context_parts.append(
            "Analyze the query and user context, then return your strategy as JSON."
        )
        context_parts.append("Use the decision logic in the system prompt above.")
        context_parts.append("")
        context_parts.append(
            "**CRITICAL:** Return ONLY the JSON object. No explanations. No markdown."
        )

        # Combine system prompt with context
        full_prompt = system_prompt + "\n\n" + "\n".join(context_parts)

        return full_prompt

    def _parse_strategy_response(
        self, response_text: str, profile_data: Optional[Dict[str, Any]]
    ) -> PlannerStrategy:
        """
        Parse LLM response into PlannerStrategy object.

        Handles markdown code blocks and validates required fields.
        Attaches the pre-fetched profile_data to the strategy.

        Args:
            response_text: Raw LLM response (should be JSON)
            profile_data: Pre-fetched profile data to attach

        Returns:
            Parsed PlannerStrategy object

        Raises:
            json.JSONDecodeError: If response is not valid JSON
            KeyError: If required fields are missing
        """
        # Strip markdown code blocks if present
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Parse JSON
        data = json.loads(text)

        # Validate required fields
        required_fields = ["recommended_tools", "fallback_tools", "reasoning"]
        for field in required_fields:
            if field not in data:
                raise KeyError(field)

        # Build strategy object
        # Note: We use the profile_data we fetched, not what LLM might return
        # (LLM shouldn't see profile_data in output since it was in input)
        return PlannerStrategy(
            recommended_tools=data["recommended_tools"],
            fallback_tools=data["fallback_tools"],
            reasoning=data["reasoning"],
            profile_data=profile_data,  # Use pre-fetched data
        )
