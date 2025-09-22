# app/agents/schemas.py
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

SchemaVersion = Literal["v1"]
Target = Literal["web", "docs", "recsys", "respond"]

class ToolCall(BaseModel):
    name: str
    args: Dict[str, Any] = Field(default_factory=dict)
    ok: Optional[bool] = None
    elapsed_ms: Optional[int] = None

class Citation(BaseModel):
    source: Literal["web", "docs", "internal"]
    ref: str
    meta: Dict[str, Any] = Field(default_factory=dict)

class BookOut(BaseModel):
    item_idx: int
    title: Optional[str] = None
    author: Optional[str] = None
    year: Optional[int] = None
    cover_id: Optional[str] = None

    class Config:
        extra = "allow"  # tolerate additional fields your builder may attach

class ChatIn(BaseModel):
    schema: SchemaVersion = "v1"
    user_text: str
    use_profile: bool = False
    force_target: Optional[Target] = None
    session_id: Optional[str] = None

class ChatOut(BaseModel):
    schema: SchemaVersion = "v1"
    target: Target
    text: str
    books: List[BookOut] = Field(default_factory=list)
    steps: List[ToolCall] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)

class RoutePlan(BaseModel):
    schema: SchemaVersion = "v1"
    target: Target
    reason: str

class TurnInput(BaseModel):
    schema: SchemaVersion = "v1"
    user_text: str
    short_history: List[Dict[str, str]] = Field(default_factory=list)
    full_history: List[Dict[str, str]] = Field(default_factory=list)
    profile_allowed: bool = False
    user_num_ratings: Optional[int] = None
    ctx: Dict[str, Any] = Field(default_factory=dict)  # e.g., {"db": db, "current_user": user}

class AgentResult(BaseModel):
    schema: SchemaVersion = "v1"
    target: Target
    text: str
    success: bool = True
    book_ids: Optional[List[int]] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    policy_version: Optional[str] = None
