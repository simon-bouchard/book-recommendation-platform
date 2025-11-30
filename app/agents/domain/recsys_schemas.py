# app/agents/domain/recsys_schemas.py
"""
Data structures for the recommendation system multi-agent pipeline.
Defines inputs and outputs for each stage: Planner, CandidateGenerator, and Curation.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PlannerInput:
    """
    Input for PlannerAgent - query analysis and strategy determination.

    The planner analyzes the user's query and decides which retrieval tools
    to use based on query type, user warmth, and profile availability.
    """

    query: str
    """User's original query/request"""

    has_als_recs_available: bool
    """Whether collaborative filtering is available (warm user with 10+ ratings)"""

    allow_profile: bool
    """Whether the planner can access user profile tools (user consent)"""

    available_retrieval_tools: list[str]
    """Names of retrieval tools available to CandidateGenerator"""


@dataclass
class PlannerStrategy:
    """
    Output from PlannerAgent - recommended strategy for candidate generation.

    Contains tool recommendations, reasoning, and any profile data gathered
    during planning. Passed to CandidateGeneratorAgent for execution.
    """

    recommended_tools: list[str]
    """Primary tools to use, ordered by preference (1-2 tools)"""

    fallback_tools: list[str]
    """Backup tools if primary underperforms (1-2 tools)"""

    reasoning: str
    """Explanation of why this strategy was chosen"""

    profile_data: Optional[dict] = None
    """
    Results from user_profile or recent_interactions tool calls.
    Example: {"user_profile": {"favorite_subjects": [12, 45, 78]}}
    """


@dataclass
class RetrievalInput:
    """
    Input for CandidateGeneratorAgent - execute strategy and gather candidates.

    Takes the strategy from PlannerAgent and executes it to retrieve 60-120
    candidate books with adaptive fallback logic.
    """

    query: str
    """Original user query"""

    strategy: PlannerStrategy
    """Strategy from PlannerAgent with tool recommendations"""

    profile_data: Optional[dict] = None
    """Profile data gathered by Planner (if any)"""


@dataclass
class ExecutionContext:
    """
    Context about how candidates were generated - passed to CurationAgent.

    Minimal info for Curator to write appropriate prose explanations.
    Curator can derive personalization and method from tools_used and profile_data.
    """

    planner_reasoning: str
    """Why Planner chose this strategy"""

    tools_used: list[str]
    """Tools executed, in order: ['als_recs'] or ['book_semantic_search', 'popular_books']"""

    profile_data: Optional[dict] = None
    """Profile data from Planner (if user profile was accessed)"""


@dataclass
class RetrievalOutput:
    """
    Output from CandidateGeneratorAgent - pool of candidate books.

    Contains 60-120 books with full metadata, plus execution context
    for CurationAgent.
    """

    candidates: list[dict]
    """
    List of 60-120 book candidates with metadata.
    Each dict contains: item_idx, title, author, subjects, tones, vibe, score
    """

    execution_context: ExecutionContext
    """Context about how candidates were generated"""

    reasoning: str
    """Explanation of execution decisions (why stopped, what was tried, etc.)"""


@dataclass
class CurationInput:
    """
    Input for CurationAgent - filter, rank, and explain recommendations.

    Takes candidate pool and produces final 8-12 recommendations with
    formatted prose explanation.
    """

    query: str
    """Original user query"""

    candidates: list[dict]
    """60-120 unfiltered candidate books from CandidateGenerator"""

    execution_context: ExecutionContext
    """How these candidates were generated"""
