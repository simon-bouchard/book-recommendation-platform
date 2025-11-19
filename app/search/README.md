# Search Module

The search module provides a flexible, adapter-based architecture for performing searches on books, with support for multiple backends. Currently, MeiliSearch is implemented; classic and semantic backends can be added via the shared adapter interface.

## Overview

The module uses a **thin orchestrator pattern**:

- **SearchEngine** — Delegates search operations to a backend adapter  
- **SearchAdapter** — Interface all search adapters must implement  
- **MeiliSearchAdapter** — Production implementation using MeiliSearch  

All business logic (filtering, sorting, pagination, highlighting, cropping) is handled in adapters.  
The orchestrator only routes requests and assembles responses.

## Architecture

SearchEngine (orchestrator)  
↓  
SearchAdapter (interface)  
├── MeiliSearchAdapter (current)  
├── ClassicSearchAdapter (planned)  
└── SemanticSearchAdapter (planned)

### Key Design Principles

- **Orchestrator is backend-agnostic**  
  The `SearchEngine` does not know how search happens — only the adapter does.

- **Adapters encapsulate logic**  
  Pagination, filters, query preprocessing, sorting, boosting, highlight/crop options — all live inside adapters.

- **Drop-in extensibility**  
  New search backends only require implementing the `SearchAdapter` interface.

- **Consistent return schema**  
  All adapters return a unified `SearchResult` object so the API layer never changes.

---

## Core Interfaces

### `SearchAdapter`

All search backends must implement the following interface:

```python
class SearchAdapter(Protocol):
    def search(
        self,
        query: str,
        subjects: Optional[list[int]] = None,
        page: int = 0
    ) -> SearchResult:
        ...
```

### `SearchResult`

Represents a normalized result, independent of backend:

```python
@dataclass
class SearchResult:
    total: int
    hits: list[dict]   # Each item is a normalized book object
    page: int
    total_pages: int
```

---

## The Orchestrator — `SearchEngine`

`SearchEngine` is intentionally tiny:

```python
class SearchEngine:
    def __init__(self, adapter: SearchAdapter):
        self.adapter = adapter

    def search(self, query: str, subjects=None, page=0):
        return self.adapter.search(query=query, subjects=subjects, page=page)
```

It does nothing except pass arguments through to the adapter.

This keeps the architecture simple, testable, and backend-agnostic.

---

## MeiliSearch Adapter (Current Implementation)

The MeiliSearch adapter handles **all real work**, including:

- building MeiliSearch filters  
- building pagination parameters  
- issuing queries  
- normalizing MeiliSearch's output into your `SearchResult`  

Example skeleton:

```python
class MeiliSearchAdapter(SearchAdapter):
    def __init__(self, client: MeiliClient):
        self.index = client.index("books")

    def search(self, query: str, subjects=None, page=0) -> SearchResult:
        meili_filter = ...
        result = self.index.search(
            query,
            {
                "filter": meili_filter,
                "page": page,
                "hitsPerPage": 20,
                "highlightPreTag": "<mark>",
                "highlightPostTag": "</mark>"
            }
        )

        return SearchResult(
            total=result["estimatedTotalHits"],
            hits=self._normalize(result["hits"]),
            page=result["page"],
            total_pages=result["totalPages"]
        )
```

---

## Why Pagination Does *Not* Belong in `SearchEngine`

Because:

- different backends support pagination differently  
- page size, cursor-based pagination, offset pagination, etc. vary  
- highlighting, boosting, and snippet cropping often require page-size-aware logic  

If `SearchEngine` handled pagination, your backend adapters would become constrained and inconsistent.

Keeping pagination inside adapters ensures:

- each backend behaves optimally  
- `SearchEngine` stays extremely thin  
- swapping in a new backend requires no API changes  

---

## Example Usage

```python
engine = SearchEngine(adapter=MeiliSearchAdapter(client))

results = engine.search(
    query="tolkien",
    subjects=[12, 27],
    page=0
)
```

---

## How to Add a New Backend

To add a new backend:

1. Create a class implementing `SearchAdapter`
2. Handle all logic (filters, scoring, pagination, normalization)
3. Return a `SearchResult`
4. Plug it into `SearchEngine`

Done.

---

## Summary

- `SearchEngine` is a thin orchestrator  
- All logic lives in backend adapters  
- Works with MeiliSearch today  
- Ready for classic or semantic search tomorrow  
- Pagination *must* stay in the adapter  
- Architecture is stable, clean, and scalable  


