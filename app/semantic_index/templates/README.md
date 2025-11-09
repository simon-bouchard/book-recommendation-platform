# Text Templates

**Purpose**: Centralized text construction logic for all embedding strategies.

## What's Here

- `text_templates.py` - Template functions for converting book metadata into embedding text
- `__init__.py` - Module exports

## Usage

```python
from app.semantic_index.templates import build_v2_subjects_text

text = build_v2_subjects_text(
    title="1984",
    author="George Orwell", 
    subjects=["dystopian society", "totalitarianism", "surveillance"]
)
# Output: "1984 — George Orwell | subjects: dystopian society, totalitarianism, surveillance"
```

## All 6 Template Functions

1. **baseline_old_text** - Title + Author + Description + OL Subjects (with description)
2. **baseline_clean_text** - Title + Author + OL Subjects (no description)
3. **v1_full_text** - Title + Author + Genre + V1 Subjects + Tones + Vibe
4. **v1_subjects_text** - Title + Author + V1 Subjects (no metadata)
5. **v2_full_text** - Title + Author + Genre + V2 Subjects + Tones + Vibe
6. **v2_subjects_text** - Title + Author + V2 Subjects (no metadata)

## Key Design Rules

- **Use names, not IDs**: Tones as strings (e.g., "dark"), genres as display names
- **Omit empty fields**: If no vibe, don't include `| vibe: ` in text
- **Consistent separator**: Always use ` | ` between sections
- **LLM-readable**: Format optimized for sentence transformer embeddings

## Used By

- **Batch builders** (`app/semantic_index/builders/`) - One-time index construction
- **Stream worker** (`app/semantic_index/embedding/worker.py`) - Continuous embedding pipeline# Text Templates for 6-Model Embedding Comparison

## Overview

This module provides centralized text template functions for all 6 embedding strategies being compared.

## File Location

**Production path**: `app/semantic_index/embedding/text_templates.py`

## The 6 Models

### Baseline Variants
1. **baseline_old** - Existing baseline with description
   - Format: `{title} — {author} | {description} | subjects: {subjects}`
   - Includes description (truncated to 500 chars)
   - Limits to 10 OL subjects

2. **baseline_clean** - New baseline without description
   - Format: `{title} — {author} | subjects: {subjects}`
   - No description (testing if it adds noise)
   - No subject limit

### V1 Variants
3. **v1_full** - Complete v1 with genre and tone names
   - Format: `{title} — {author} | genre: {genre_name} | subjects: {subjects} | tones: {tones} | vibe: {vibe}`
   - **IMPORTANT**: This corrects bugs in existing v1 index (used tone IDs, missing genre)
   - **ACTION REQUIRED**: V1 index must be REBUILT before comparison

4. **v1_subjects** - V1 subjects only
   - Format: `{title} — {author} | subjects: {subjects}`
   - Tests if v1 subjects alone beat full metadata

### V2 Variants
5. **v2_full** - Complete v2 with genre and tone names
   - Format: `{title} — {author} | genre: {genre_name} | subjects: {subjects} | tones: {tones} | vibe: {vibe}`
   - Uses v2 ontology (36 tones in 6 buckets)
   - Built from accumulator batches

6. **v2_subjects** - V2 subjects only
   - Format: `{title} — {author} | subjects: {subjects}`
   - **KEY COMPARISON**: v2_subjects vs v2_full to see if metadata is noise

## Critical Design Decisions

### Use Names, Not IDs
- **Tone**: Use slug names (e.g., "dark", "fast-paced"), NOT integer IDs
- **Genre**: Use display names (e.g., "Science Fiction"), NOT slugs or IDs
- **Rationale**: LLMs can't understand integer IDs in embeddings

### Empty Field Handling
- Empty tones/vibe are OMITTED (not included in text)
- Genre is REQUIRED for "full" variants
- Subjects should not be empty (but no hard requirement)

## Next Steps

### Phase 1 Complete ✅
- Text templates created
- All 6 formats defined
- Documentation written

### Phase 2: Build Missing Indexes
1. **CRITICAL**: Rebuild V1-Full index with corrected format
   - Current v1 index uses tone IDs (wrong)
   - Current v1 index missing genre (wrong)
   - Use `build_v1_full_text()` template
   
2. Build baseline-clean index
   - Use `build_baseline_clean_text()` template
   
3. Build v1-subjects index
   - Use `build_v1_subjects_text()` template
   
4. Build v2-subjects index (when v2 enrichment complete)
   - Use `build_v2_subjects_text()` template
   
5. Build v2-full index (when v2 enrichment complete)
   - Use `build_v2_full_text()` template
   - Build from accumulator batches

### Phase 3: Run Comparison
- Multi-model comparison HTML report
- Manual evaluation of results
- Statistical analysis

## Implementation Notes

### Tone/Genre Resolution Required
For v1-full and v2-full builds, you'll need to:
1. Load ontology CSVs (tones, genres)
2. Map IDs to display names:
   - `tone_id → tone_slug` (e.g., 5 → "dark")
   - `genre_id → genre_name` (e.g., 3 → "Historical Fiction")
3. Pass resolved names to template functions

### Data Sources
- **baseline-old/clean**: Books table + OL subjects (raw)
- **v1-subjects**: `book_llm_subjects` WHERE `tags_version='v1'`
- **v1-full**: Above + `book_tones` + `book_genres` + `book_vibes` (all v1)
- **v2-subjects**: `book_llm_subjects` WHERE `tags_version='v2'`
- **v2-full**: Read from accumulator `.npz` batches (already embedded with full text)

## Example Usage

```python
from app.semantic_index.embedding.text_templates import (
    build_baseline_clean_text,
    build_v1_subjects_text,
    build_v2_full_text
)

# Baseline-clean
text = build_baseline_clean_text(
    title="1984",
    author="George Orwell",
    ol_subjects=["dystopian fiction", "totalitarianism", "surveillance"]
)
# Output: "1984 — George Orwell | subjects: dystopian fiction, totalitarianism, surveillance"

# V1-subjects
text = build_v1_subjects_text(
    title="1984",
    author="George Orwell",
    subjects=["dystopian society", "government surveillance", "thought control"]
)
# Output: "1984 — George Orwell | subjects: dystopian society, government surveillance, thought control"

# V2-full (with resolved names)
text = build_v2_full_text(
    title="1984",
    author="George Orwell",
    genre_name="Science Fiction",  # Resolved from genre_id
    subjects=["totalitarian state", "dystopian future", "propaganda"],
    tone_names=["dark", "cerebral", "disturbing"],  # Resolved from tone_ids
    vibe="Chilling dystopian vision of totalitarian surveillance state"
)
# Output: "1984 — George Orwell | genre: Science Fiction | subjects: totalitarian state, dystopian future, propaganda | tones: dark, cerebral, disturbing | vibe: Chilling dystopian vision of totalitarian surveillance state"
```

## Validation

Each template function:
- Returns a non-empty string
- Uses consistent separators (` | `)
- Omits empty optional fields
- Maintains clean, readable format

Use `validate_text_not_empty()` utility to check outputs.
