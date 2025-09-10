# app/agents/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional

class ChatIn(BaseModel):
    message: str
    use_profile: bool = True
    restrict_to_catalog: bool = True

class BookOut(BaseModel):
    item_idx: int
    title: Optional[str] = None
    author: Optional[str] = None
    year: Optional[int] = None
    cover_id: Optional[str] = None
    
class ChatOut(BaseModel):
    reply: str
    books: List[BookOut] = Field(default_factory=list)