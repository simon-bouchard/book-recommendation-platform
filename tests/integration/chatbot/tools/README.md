# Tool Integration Tests

Integration tests that verify all tools work correctly with real dependencies (database, external APIs, help files).

## Purpose

These tests ensure that:
1. Tools don't crash when called with typical inputs
2. Tools return expected output structure
3. Tools handle parameter validation correctly
4. Tools handle error conditions appropriately

## Running Tests

```bash
# Run all tool integration tests
pytest tests/integration/chatbot/tools/

# Run specific test file
pytest tests/integration/chatbot/tools/test_retrieval_tools.py
pytest tests/integration/chatbot/tools/test_web_tools.py
pytest tests/integration/chatbot/tools/test_docs_tools.py

# Run specific test class
pytest tests/integration/chatbot/tools/test_retrieval_tools.py::TestPopularBooks

# Run specific test
pytest tests/integration/chatbot/tools/test_retrieval_tools.py::TestPopularBooks::test_returns_requested_number
```

## Requirements

### Retrieval Tools
- Production database accessible via DATABASE_URL
- Database contains book and subject data
- Semantic search FAISS index at `models/artifacts/semantic_indexes/enriched_v2/`
- User with ID 278859 exists for ALS recommendation tests

### Web Tools
- Internet connection for external API calls
- DuckDuckGo search API accessible
- Wikipedia API accessible
- OpenLibrary API accessible

### Documentation Tools
- Help documentation files in `docs/help/`
- Help manifest JSON file exists

## Test Coverage

### Retrieval Tools (test_retrieval_tools.py)

**Popular Books**
- Returns requested number of books
- Returns standardized fields
- Clamps top_k to [1, 500]
- Returns error without database

**Semantic Search**
- Returns results for query
- Returns standardized fields
- Clamps top_k to [1, 500]

**Subject ID Search**
- Returns results for phrases
- Candidate structure validation
- Clamps top_k to [1, 10]
- Returns error without database

**Subject Hybrid**
- Returns results for subject indices
- Returns standardized fields
- Uses user's favorite subjects when empty list provided
- Clamps top_k to [1, 500]
- Clamps subject_weight to [0, 1]
- Returns error without database
- Returns error without subjects and no user

**ALS Recommendations**
- Returns results for authenticated user
- Returns standardized fields
- Clamps top_k to [1, 500]
- Returns error without authentication

### Web Tools (test_web_tools.py)

**Web Search**
- Returns results for query
- Deduplicates repeat queries

**Wikipedia Lookup**
- Returns article for topic
- Deduplicates repeat queries

**OpenLibrary Search**
- Returns books for query
- Deduplicates repeat queries

**OpenLibrary Work**
- Returns work details
- Handles work key normalization
- Deduplicates repeat queries

### Documentation Tools (test_docs_tools.py)

**Help Manifest**
- Returns manifest dictionary
- Manifest has expected structure

**Help Read**
- Reads document by alias
- Returns error for invalid document

## Test Organization

Tests are organized by tool category:
- `test_retrieval_tools.py` - Internal recommendation tools
- `test_web_tools.py` - External web/API tools
- `test_docs_tools.py` - Documentation/help tools

Each file contains test classes for individual tools with tests for specific behaviors.
