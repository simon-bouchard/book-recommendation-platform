# Search Module

The search module provides a flexible, adapter-based architecture for performing book searches with support for multiple search backends (currently MeiliSearch, with support for classic and semantic search planned).

## Overview

The module follows a **thin orchestrator pattern** with delegated business logic:

- **SearchEngine** - Orchestrator that routes search requests to appropriate adapters
- **SearchAdapter** - Base interface for pluggable search implementations
- **MeiliSearchAdapter** - Current production adapter using MeiliSearch as the search backend

All search logic (filtering, sorting, pagination) is pushed down to the adapter level, keeping the orchestrator simple and enabling different optimization strategies per backend.

## Architecture

```
SearchEngine (orchestrator)
    ↓
SearchAdapter (interface)
    ├── MeiliSearchAdapter (MeiliSearch backend)
    ├── ClassicSearchAdapter (planned)
    └── SemanticSearchAdapter (planned)
```

### Key Design Principles

1. **No logic at orchestrator level** - SearchEngine only handles delegation and error handling
2. **All business logic in adapters** - Each adapter fully owns filtering, sorting, and pagination
3. **Easy extensibility** - Add new search modes by implementing SearchAdapter interface
4. **Dynamic mode discovery** - Adapters report availability; orchestrator only exposes available modes

## Components

### Models (`models.py`)

Core data structures:

- **SearchMode** (Enum): Supported search backends
  - `MEILI` - MeiliSearch (production)
  - `CLASSIC` - Full-text with fallback ranking (planned)
  - `SEMANTIC` - Vector-based semantic search (planned)

- **SearchRequest**: Input contract
  ```python
  {
    query: str,              # Search query text
    mode: SearchMode,        # Which backend to use
    filters: Dict,           # Key-value filters (e.g., subject_ids)
    sort: Optional[str],     # Sort spec (e.g., "year:desc", "bayes_pop:desc")
    page: int,              # Zero-indexed page number
    page_size: int          # Results per page
  }
  ```

- **SearchResult**: Individual search result
  ```python
  {
    item_idx: int,          # Book identifier
    title: str,
    author: str,
    cover_id: Optional[int],
    _score: Optional[float] # Engine-specific relevance score
  }
  ```

- **SearchResponse**: Output contract
  ```python
  {
    results: List[SearchResult],
    total: int,             # Total matching results
    page: int,
    page_size: int
  }
  ```

### SearchEngine (`engine.py`)

Main entry point. Routes requests to adapters and provides health discovery.

```python
from app.search.engine import SearchEngine
from app.search.models import SearchRequest, SearchMode

engine = SearchEngine()

# Simple search
request = SearchRequest(
    query="python programming",
    mode=SearchMode.MEILI,
    page=0,
    page_size=50
)
response = engine.search(request)

# With filters and sorting
request = SearchRequest(
    query="fiction",
    mode=SearchMode.MEILI,
    filters={"subject_ids": [1, 2, 3]},
    sort="bayes_pop:desc",
    page=0,
    page_size=50
)
response = engine.search(request)

# Check available modes
available = engine.get_available_modes()  # [SearchMode.MEILI, ...]
```

**Key Methods:**

- `search(request: SearchRequest) -> SearchResponse` - Execute a search
- `get_available_modes() -> List[SearchMode]` - Discover available search backends

### Adapters

#### Base Interface (`adapters/base.py`)

```python
class SearchAdapter(ABC):
    @abstractmethod
    def search(self, request: SearchRequest) -> Tuple[List[SearchResult], int]:
        """Return (results, total_count)"""
        pass
    
    @abstractmethod
    def is_available() -> bool:
        """Health check - is this adapter ready?"""
        pass
    
    @property
    @abstractmethod
    def mode(self) -> SearchMode:
        """Which search mode does this handle?"""
        pass
```

#### MeiliSearch Adapter (`adapters/meili.py`)

Production adapter using MeiliSearch for full-text search with advanced features.

**Features:**
- Full-text search with BM25 ranking
- Filtering using MeiliSearch filter syntax
- Sorting (e.g., by popularity, year)
- Pagination via limit/offset

**Configuration:**
- Connects to MeiliSearch instance at `http://localhost:7700`
- Master key from environment variable `MEILI_MASTER_KEY`
- Indexes book data in `books` index

**Filter Syntax:**

Filters are passed as a dictionary and converted to MeiliSearch filter strings:

```python
# Single value filter
{"subject_ids": 42}
# → 'subject_ids = "42"'

# Multiple value filter (OR)
{"subject_ids": [1, 2, 3]}
# → '(subject_ids = "1" OR subject_ids = "2" OR subject_ids = "3")'
```

**Sorting:**

Sort specifications follow MeiliSearch format: `"field:direction"` where direction is `asc` or `desc`.

Examples:
- `"bayes_pop:desc"` - Bayesian popularity (descending)
- `"year:desc"` - Publication year (newest first)
- `"title:asc"` - Alphabetical by title

## Usage

### Basic Search

```python
from app.search.search_utils import get_search_results

results, has_next, total = get_search_results(
    query="science fiction",
    subject="1",
    page=0,
    page_size=50,
    db=db_connection
)
```

### Advanced Filtering

```python
from app.search.engine import SearchEngine
from app.search.models import SearchRequest, SearchMode

engine = SearchEngine()

# Multi-subject filter (OR logic)
request = SearchRequest(
    query="mystery",
    mode=SearchMode.MEILI,
    filters={"subject_ids": [10, 11, 12]},  # Any of these subjects
    sort="bayes_pop:desc",
    page=0,
    page_size=20
)

response = engine.search(request)
for result in response.results:
    print(f"{result.title} by {result.author}")
```

## Adding a New Search Adapter

To add support for a new search backend:

1. **Create the adapter class** that inherits from `SearchAdapter`
2. **Implement required methods**: `search()`, `is_available()`, and `mode` property
3. **Register in SearchEngine** by adding to `_initialize_adapters()` dict

**Example: Adding a Classic Search Adapter**

```python
# app/search/adapters/classic.py
from .base import SearchAdapter
from ..models import SearchRequest, SearchResult, SearchMode
from typing import List, Tuple

class ClassicSearchAdapter(SearchAdapter):
    def search(self, request: SearchRequest) -> Tuple[List[SearchResult], int]:
        # Implement classic full-text search logic
        # Convert request to your backend's query format
        # Execute query and convert results to SearchResult objects
        pass
    
    def is_available(self) -> bool:
        # Check if your backend is healthy
        pass
    
    @property
    def mode(self) -> SearchMode:
        return SearchMode.CLASSIC
```

Then register in `engine.py`:

```python
def _initialize_adapters(self) -> Dict[SearchMode, SearchAdapter]:
    return {
        SearchMode.MEILI: MeiliSearchAdapter(),
        SearchMode.CLASSIC: ClassicSearchAdapter(),  # NEW
    }
```

## Environment Variables

- `MEILI_MASTER_KEY` - MeiliSearch authentication key

## Dependencies

- `meilisearch` - Python client for MeiliSearch
- `pydantic` - Data validation and modeling
- `python-dotenv` - Environment variable management

## Testing

To test the search module:

```python
# Check available modes
from app.search.engine import SearchEngine
engine = SearchEngine()
print(engine.get_available_modes())  # Should include at least SearchMode.MEILI

# Try a search
from app.search.models import SearchRequest, SearchMode
request = SearchRequest(query="test", mode=SearchMode.MEILI, page=0, page_size=10)
response = engine.search(request)
print(f"Found {response.total} results")
```

## Troubleshooting

### No search results
- Verify MeiliSearch is running at `http://localhost:7700`
- Check that the `books` index exists and is populated
- Verify filter syntax matches your data structure

### "Unsupported search mode" error
- Check that `SearchMode` enum includes the requested mode
- Ensure the adapter for that mode is registered in `SearchEngine._initialize_adapters()`
- Call `engine.get_available_modes()` to see what's available

### MeiliSearch adapter returns `is_available() = False`
- Verify MeiliSearch container is running
- Check `MEILI_MASTER_KEY` environment variable is set
- Verify network connectivity to MeiliSearch instance
