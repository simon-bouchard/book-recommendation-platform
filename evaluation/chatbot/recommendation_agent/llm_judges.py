# evaluation/chatbot/recommendation_agent/llm_judges.py
"""
LLM-as-judge functions for semantic validation in recommendation agent evaluation.
Provides judges for genre matching, personalization prose, prose reasoning quality, and query relevance.
"""

import json
from typing import Any, Dict, List, Optional

from app.table_models import Book


def llm_judge_prose_reasoning(
    response_text: str, judge_llm, query_type: str = "specific"
) -> Dict[str, Any]:
    """
    Use LLM-as-judge to verify prose quality, with criteria branching on query type.

    For specific queries (descriptive/genre), prose must explain why each book fits
    the request — referencing the query's themes, tone, or constraints.

    For vague queries ("recommend me something"), the user has given no constraints
    to match against, so prose must instead be vivid and specific enough for the
    user to self-select. A generic one-line summary fails in both modes.

    Args:
        response_text: Curation agent's prose explanation
        judge_llm: LLM instance for judging
        query_type: "specific" (descriptive/genre) or "vague" (open-ended).
            Defaults to "specific" for backward compatibility.

    Returns:
        Dict with verdict, passed status, reasoning, and issues found
    """
    if query_type == "vague":
        criteria = """QUERY TYPE: VAGUE (open-ended request, no specific constraints)

The user gave an open-ended request so there is nothing to match against.
Instead, the prose must help the user decide whether each book sounds appealing.

CRITERIA FOR PASSING:
- Each book is described vividly enough for a reader to self-select
- Descriptions convey tone, pacing, or who the book is for — not just plot
- No book is dismissed in a single generic sentence ("A classic thriller set in Paris")

EXAMPLES:
BAD  — "A gripping thriller set in Paris." (too generic — tells the user nothing)
BAD  — "A classic novel exploring themes of family." (meaningless without specifics)
GOOD — "A slow-burn psychological thriller with an unreliable narrator — ideal if you enjoy stories where you're never quite sure what's real."
GOOD — "Dense, idea-driven sci-fi that rewards patient readers; big-picture philosophy wrapped in an adventure plot.\""""
    else:
        criteria = """QUERY TYPE: SPECIFIC (descriptive or genre query with stated constraints)

The user made a specific request with themes, tone, genre, or mood. The prose must
explain why each book fits that request — not just describe what the book is about.

CRITERIA FOR PASSING:
- Most citations lead with fit, then use description to support it
- Prose references at least some of the query's stated attributes (genre, mood, themes, tone)
- Generic plot summaries that ignore the query do not pass

EXAMPLES:
BAD  — "A fantasy novel about a young wizard who attends a magical school." (describes the book, ignores the query)
BAD  — "A classic mystery." (too vague — says nothing about query fit)
GOOD — "Matches your request for atmospheric mystery — the isolated setting and unreliable narrator create exactly the psychological tension you asked for."
GOOD — "Fits the dark, brooding tone you described, with morally grey characters and a world that feels genuinely threatening.\""""

    judge_prompt = f"""You are evaluating a book recommendation system's prose quality.

AGENT'S PROSE RESPONSE:
{response_text}

{criteria}

Return JSON:
{{
    "passed": true/false,
    "verdict": "summary of findings",
    "reasoning": "detailed explanation of whether prose meets the criteria above",
    "issues": ["list of problems if any"]
}}"""

    try:
        response = judge_llm.invoke([{"role": "user", "content": judge_prompt}])

        content = response.content if hasattr(response, "content") else str(response)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        content = content.strip()

        result = json.loads(content)

        result.setdefault("passed", False)
        result.setdefault("verdict", "Unknown")
        result.setdefault("reasoning", "No reasoning provided")
        result.setdefault("issues", [])

        return result

    except Exception as e:
        return {
            "passed": False,
            "verdict": "Judge evaluation failed",
            "reasoning": f"Error during LLM judge evaluation: {str(e)}",
            "issues": [str(e)],
            "error": str(e),
        }


def llm_judge_query_relevance(
    response_text: str, query: str, judge_llm, tools_used: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Use LLM-as-judge to verify prose addresses and relates to the user's query.

    Args:
        response_text: Curation agent's prose explanation
        query: Original user query
        judge_llm: LLM instance for judging
        tools_used: List of tools used to generate recommendations (e.g., ['als_recs'])

    Returns:
        Dict with verdict, passed status, reasoning, and issues found
    """
    judge_prompt = f"""You are evaluating a book recommendation system's query relevance.

USER'S QUERY:
{query}

TOOLS USED TO GENERATE RECOMMENDATIONS:
{tools_used if tools_used else "Unknown"}

AGENT'S PROSE RESPONSE:
{response_text}

EVALUATION CRITERIA:
The prose should appropriately acknowledge how recommendations were generated.

**For ALS (collaborative filtering) - tools contain "als_recs":**
- Prose SHOULD mention: "based on your reading history", "books you've enjoyed", "given your past ratings", "personalized for you", or similar
- May also reference query if specific, but personalization mention is primary

**For profile/subject tools - tools contain "subject_hybrid_pool":**
- Prose SHOULD mention user's interests/preferences: "your interest in [genre]", "based on your favorite genres", "genres you enjoy"
- Should acknowledge that recommendations reflect user's stated preferences

**For semantic search - tools contain "book_semantic_search":**
- Prose SHOULD reference query elements (genre, themes, tone, atmosphere) if query is specific
- For vague queries with semantic search, prose should still explain book qualities/characteristics

**For popular books - tools contain "popular_books":**
- Prose SHOULD mention popularity: "widely loved", "popular books", "bestsellers", "acclaimed", "highly rated"
- Appropriate for cold users or when personalization isn't available

**General principles:**
- Vague queries (e.g., "recommend something") require method acknowledgment, not query element matching
- Specific queries (e.g., "dark mystery") require both method acknowledgment AND query element references
- The prose should make it clear HOW the recommendations were chosen

Return JSON:
{{
    "passed": true/false,
    "verdict": "summary of findings",
    "reasoning": "detailed explanation of how well prose acknowledges the recommendation method and query",
    "issues": ["list of disconnects if any"]
}}

CRITERIA FOR PASSING:
- Prose appropriately acknowledges the tool/method used to generate recommendations
- If query is specific AND semantic search used, also references query elements
- Clear connection between recommendation method and prose explanation
- User understands WHY these books were chosen
"""

    try:
        response = judge_llm.invoke([{"role": "user", "content": judge_prompt}])

        content = response.content if hasattr(response, "content") else str(response)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        content = content.strip()

        result = json.loads(content)

        result.setdefault("passed", False)
        result.setdefault("verdict", "Unknown")
        result.setdefault("reasoning", "No reasoning provided")
        result.setdefault("issues", [])

        return result

    except Exception as e:
        return {
            "passed": False,
            "verdict": "Judge evaluation failed",
            "reasoning": f"Error during LLM judge evaluation: {str(e)}",
            "issues": [str(e)],
            "error": str(e),
        }


def llm_judge_genre_match(books: List[Any], expected_genre: str, db, judge_llm) -> Dict[str, Any]:
    """
    Use LLM-as-judge to determine if recommended books match expected genre.

    Args:
        books: List of recommended BookRecommendation objects
        expected_genre: Genre that books should match (e.g., "fantasy", "historical fiction")
        db: Database session to fetch book details
        judge_llm: LLM instance for judging (reuse agent's LLM)

    Returns:
        Dict with verdict, passed status, reasoning, and violating books
    """
    book_ids = [book.item_idx for book in books]
    items = db.query(Book).filter(Book.item_idx.in_(book_ids)).all()

    book_details = []
    for item in items:
        author_name = item.author.name if item.author else "Unknown"

        subject_names = [bs.subject.subject for bs in item.subjects] if item.subjects else []

        book_details.append(
            {
                "id": item.item_idx,
                "title": item.title,
                "author": author_name,
                "subjects": subject_names,
                "description": item.description[:200] if item.description else "No description",
            }
        )

    judge_prompt = f"""You are evaluating a book recommendation system's genre accuracy.

EXPECTED GENRE: {expected_genre}

RECOMMENDED BOOKS:
{json.dumps(book_details, indent=2)}

TASK:
Determine if these books match the expected genre "{expected_genre}".
- Check book titles, subjects, and descriptions
- Allow reasonable subgenres (e.g., "urban fantasy" counts as "fantasy")
- Flag books that clearly belong to different genres

Return JSON:
{{
    "passed": true/false,
    "verdict": "brief summary of findings",
    "reasoning": "detailed explanation of genre matching",
    "violating_books": [
        {{"id": 123, "title": "Book Title", "reason": "why it doesn't match"}}
    ],
    "match_count": X,
    "total_count": Y
}}

CRITERIA FOR PASSING:
- At least 80% of books should match the expected genre
- Minor mismatches acceptable (e.g., mystery/thriller overlap)
- Clear genre violations should be flagged
"""

    try:
        response = judge_llm.invoke([{"role": "user", "content": judge_prompt}])

        content = response.content if hasattr(response, "content") else str(response)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        content = content.strip()

        result = json.loads(content)

        result.setdefault("passed", False)
        result.setdefault("verdict", "Unknown")
        result.setdefault("reasoning", "No reasoning provided")
        result.setdefault("violating_books", [])

        return result

    except Exception as e:
        return {
            "passed": False,
            "verdict": "Judge evaluation failed",
            "reasoning": f"Error during LLM judge evaluation: {str(e)}",
            "violating_books": [],
            "error": str(e),
        }


def llm_judge_negative_constraint_filtering(
    books: List[Any],
    negative_constraints: List[str],
    db,
    judge_llm,
) -> Dict[str, Any]:
    """
    Use LLM-as-judge to verify no selected book matches an excluded constraint.

    Args:
        books: List of selected BookRecommendation objects
        negative_constraints: Terms the user explicitly excluded (e.g., ["cozy", "cozy mystery"])
        db: Database session to fetch book details
        judge_llm: LLM instance for judging

    Returns:
        Dict with verdict, passed status, reasoning, violating_books, and violation_count
    """
    book_ids = [book.item_idx for book in books]
    items = db.query(Book).filter(Book.item_idx.in_(book_ids)).all()

    book_details = []
    for item in items:
        author_name = item.author.name if item.author else "Unknown"
        subject_names = [bs.subject.subject for bs in item.subjects] if item.subjects else []
        book_details.append(
            {
                "id": item.item_idx,
                "title": item.title,
                "author": author_name,
                "subjects": subject_names,
                "description": item.description[:200] if item.description else "No description",
            }
        )

    constraints_str = ", ".join(f'"{c}"' for c in negative_constraints)

    judge_prompt = f"""You are evaluating a book recommendation system's negative constraint filtering.

USER EXCLUDED THESE TERMS: {constraints_str}

SELECTED BOOKS:
{json.dumps(book_details, indent=2)}

TASK:
Determine whether any of these books clearly match the excluded terms.
Check titles, subjects, and descriptions for evidence of the constraint.
A book "matches" only if there is clear evidence — do not penalise tangential overlap.

Return JSON:
{{
    "passed": true/false,
    "verdict": "brief summary of findings",
    "reasoning": "detailed explanation of what you checked and why each decision was made",
    "violating_books": [
        {{"id": 123, "title": "Book Title", "reason": "why it matches the excluded constraint"}}
    ],
    "violation_count": 0
}}

CRITERIA FOR PASSING:
- No selected book clearly matches any of the excluded terms
- Minor thematic overlap is acceptable; only flag clear matches
"""

    try:
        response = judge_llm.invoke([{"role": "user", "content": judge_prompt}])

        content = response.content if hasattr(response, "content") else str(response)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        content = content.strip()

        result = json.loads(content)

        result.setdefault("passed", False)
        result.setdefault("verdict", "Unknown")
        result.setdefault("reasoning", "No reasoning provided")
        result.setdefault("violating_books", [])
        result.setdefault("violation_count", len(result["violating_books"]))

        return result

    except Exception as e:
        return {
            "passed": False,
            "verdict": "Judge evaluation failed",
            "reasoning": f"Error during LLM judge evaluation: {str(e)}",
            "violating_books": [],
            "violation_count": 0,
            "error": str(e),
        }


def llm_judge_personalization_prose(
    response_text: str, execution_context, judge_llm, expect_no_personalization: bool = False
) -> Dict[str, Any]:
    """
    Use LLM-as-judge to verify curation prose correctly reflects personalization context.

    Validates that prose appropriately mentions (or doesn't mention) personalization
    based on how recommendations were generated.

    Args:
        response_text: Curation agent's prose explanation
        execution_context: ExecutionContext showing how recommendations were made
        judge_llm: LLM instance for judging (reuse agent's LLM)
        expect_no_personalization: If True, prose should NOT claim personalization

    Returns:
        Dict with verdict, passed status, reasoning, and issues found
    """
    judge_prompt = f"""You are evaluating a book recommendation system's prose explanation.

CONTEXT PROVIDED TO AGENT:
- Tools used: {execution_context.tools_used}
- Profile data available: {execution_context.profile_data is not None}
- Planner reasoning: {execution_context.planner_reasoning}
"""

    if execution_context.profile_data:
        judge_prompt += f"\n- Profile data: {json.dumps(execution_context.profile_data, indent=2)}"

    judge_prompt += f"""

AGENT'S PROSE RESPONSE:
{response_text}

"""

    if expect_no_personalization:
        judge_prompt += """EVALUATION CRITERIA:
- The recommendations were NOT personalized (no ALS, no profile data used)
- The prose should NOT claim personalization
- Check for false claims like "based on your reading history", "given your preferences", etc.
- Prose should focus on query matching or general quality instead

Return JSON:
{
    "passed": true/false,
    "verdict": "summary of findings",
    "reasoning": "detailed explanation of whether prose correctly avoids personalization claims",
    "issues": ["list of any false personalization claims found"]
}

CRITERIA FOR PASSING:
- Prose does NOT falsely claim personalization
- Focus is on query matching or book quality, not user history/preferences
"""

    elif "als_recs" in execution_context.tools_used:
        judge_prompt += """EVALUATION CRITERIA:
- The recommendations were based on collaborative filtering (user's reading history via ALS)
- The prose SHOULD mention this personalization
- Look for phrases like:
  * "based on your reading history"
  * "similar to books you've enjoyed"
  * "given your past ratings"
  * "personalized for you"
  * Other natural references to user's reading patterns

Return JSON:
{
    "passed": true/false,
    "verdict": "summary of findings",
    "reasoning": "detailed explanation of whether prose correctly indicates personalization",
    "issues": ["list of any problems found"]
}

CRITERIA FOR PASSING:
- Prose naturally mentions that recommendations are based on user's reading history
- References can be subtle (e.g., "these match your taste") or explicit
"""

    elif execution_context.profile_data:
        profile = execution_context.profile_data

        if "user_profile" in profile and profile["user_profile"].get("favorite_subjects"):
            favorite_subjects = profile["user_profile"]["favorite_subjects"]
            favorite_genres = profile["user_profile"].get("favorite_genres", [])

            judge_prompt += f"""EVALUATION CRITERIA:
- The recommendations were based on user's favorite subjects: {favorite_subjects}
- User's favorite genres: {favorite_genres}
- The prose SHOULD reference these interests or preferences
- Look for phrases like:
  * "given your interest in [genre]"
  * "based on your favorite genres"
  * "matching your preferences"
  * References to specific genres the user likes
  * "genres you enjoy"

Return JSON:
{{
    "passed": true/false,
    "verdict": "summary of findings",
    "reasoning": "detailed explanation of whether prose mentions user's interests",
    "issues": ["list of any problems found"]
}}

CRITERIA FOR PASSING:
- Prose mentions user's interests, favorite genres, or preferences
- Natural acknowledgment that recommendations reflect user's tastes
"""

        elif "recent_interactions" in profile:
            recent_books = profile["recent_interactions"][:3]
            recent_titles = [b["title"] for b in recent_books]

            judge_prompt += f"""EVALUATION CRITERIA:
- The recommendations considered user's recent reads: {recent_titles}
- The prose SHOULD acknowledge these recent books or reading patterns
- Look for phrases like:
  * "similar to [book title]"
  * "since you enjoyed [book]"
  * "based on your recent reading"
  * References to patterns in recent books
  * "books like the ones you've been reading"

Return JSON:
{{
    "passed": true/false,
    "verdict": "summary of findings",
    "reasoning": "detailed explanation of whether prose references recent reading",
    "issues": ["list of any problems found"]
}}

CRITERIA FOR PASSING:
- Prose acknowledges user's recent reads or reading patterns
- May mention specific book titles or general patterns
"""

    else:
        judge_prompt += """EVALUATION CRITERIA:
- The recommendations were based on query matching (no personalization)
- The prose should focus on how books match the query
- Should NOT falsely claim personalization

Return JSON:
{
    "passed": true/false,
    "verdict": "summary of findings",
    "reasoning": "detailed explanation of prose appropriateness",
    "issues": ["list of any problems found"]
}

CRITERIA FOR PASSING:
- Prose focuses on query matching or book quality
- No false personalization claims
"""

    try:
        response = judge_llm.invoke([{"role": "user", "content": judge_prompt}])

        content = response.content if hasattr(response, "content") else str(response)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        content = content.strip()

        result = json.loads(content)

        result.setdefault("passed", False)
        result.setdefault("verdict", "Unknown")
        result.setdefault("reasoning", "No reasoning provided")
        result.setdefault("issues", [])

        return result

    except Exception as e:
        return {
            "passed": False,
            "verdict": "Judge evaluation failed",
            "reasoning": f"Error during LLM judge evaluation: {str(e)}",
            "issues": [str(e)],
            "error": str(e),
        }
