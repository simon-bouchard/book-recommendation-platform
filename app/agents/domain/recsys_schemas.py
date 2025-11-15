# app/agents/domain/recsys_schemas.py
"""
Data structures for the recommendation system multi-agent pipeline.
Defines inputs and outputs for each stage: Planner, CandidateGenerator, and Curation.
"""
from dataclasses import dataclass, field
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
    
    negative_constraints: Optional[list[str]] = None
    """
    Detected negative constraints from query (e.g., ["vampires", "romance"]).
    Logged for analysis; CandidateGenerator ignores them, Curator filters them.
    """


@dataclass
class ToolExecution:
    """
    Record of a single tool execution during candidate generation.
    
    Tracks which tool was called, with what parameters, and the result quality.
    Used for execution context passed to CurationAgent.
    """
    tool_name: str
    """Name of the tool that was executed"""
    
    arguments: dict
    """Parameters passed to the tool"""
    
    book_count: int
    """Number of books returned by this tool"""
    
    relevance_assessment: str
    """Quality assessment: 'good match' | 'off-target' | 'mixed'"""
    
    order: int
    """Execution sequence number (1st, 2nd, 3rd, etc.)"""


@dataclass
class CandidateGeneratorInput:
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
class CandidateGeneratorOutput:
    """
    Output from CandidateGeneratorAgent - pool of candidate books.
    
    Contains 60-120 books with full metadata, plus execution history
    for transparency and debugging.
    """
    candidates: list[dict]
    """
    List of 60-120 book candidates with metadata.
    Each dict contains: item_idx, title, author, subjects, tones, description, score
    """
    
    tool_executions: list[ToolExecution]
    """History of tool calls made during candidate generation"""
    
    reasoning: str
    """Explanation of execution decisions (why stopped, what was tried, etc.)"""


@dataclass
class ExecutionContext:
    """
    Context about how candidates were generated - passed to CurationAgent.
    
    Provides transparency about the retrieval process so the curator can
    explain recommendations appropriately (e.g., "personalized for you" vs
    "popular books matching your interests").
    """
    planner_reasoning: str
    """Why the planner chose this strategy"""
    
    tool_executions: list[ToolExecution]
    """What tools were called and in what order"""
    
    profile_data: Optional[dict] = None
    """Whether profile data was used in retrieval"""


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
