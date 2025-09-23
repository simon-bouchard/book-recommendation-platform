# app/agents/schemas.py
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, AliasChoices

SchemaVersion = Literal["v1"]
Target = Literal["web", "docs", "recsys", "respond"]

def _version_field():
    # Accept both "schema" and "version" on input, serialize as "schema"
    return Field(
        default="v1",
        validation_alias=AliasChoices("schema", "version"),
        serialization_alias="schema",
    )

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
    version: SchemaVersion = _version_field()
    user_text: str
    use_profile: bool = False
    force_target: Optional[Target] = None
    session_id: Optional[str] = None

class ChatOut(BaseModel):
    version: SchemaVersion = _version_field()
    target: Target
    text: str
    books: List[BookOut] = Field(default_factory=list)
    steps: List[ToolCall] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)

class RoutePlan(BaseModel):
    version: SchemaVersion = _version_field()
    target: Target
    reason: str

class TurnInput(BaseModel):
    version: SchemaVersion = _version_field()
    user_text: str
    full_history: List[Dict[str, str]] = Field(default_factory=list)
    profile_allowed: bool = False
    user_num_ratings: Optional[int] = None
    ctx: Dict[str, Any] = Field(default_factory=dict)  # e.g., {"db": db, "current_user": user}

class AgentResult(BaseModel):
    version: SchemaVersion = _version_field()
    target: Target
    text: str
    success: bool = True
    book_ids: Optional[List[int]] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    policy_version: Optional[str] = None
