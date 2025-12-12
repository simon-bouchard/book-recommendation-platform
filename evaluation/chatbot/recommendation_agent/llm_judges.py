# evaluation/chatbot/recommendation_agent/llm_judges.py
"""
LLM-as-judge functions for semantic validation in recommendation agent evaluation.
Provides judges for genre matching and personalization prose quality.
"""

import json
from typing import Dict, List, Any, Optional


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
    from app.table_models import Book

    # Get book details from database
    book_ids = [book.item_idx for book in books]
    items = db.query(Book).filter(Book.item_idx.in_(book_ids)).all()

    book_details = []
    for item in items:
        author_name = item.author.name if item.author else "Unknown"

        # Extract subject names from BookSubject relationship
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

    # Build judge prompt
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

        # Parse JSON response (handle markdown code blocks)
        content = response.content if hasattr(response, "content") else str(response)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        content = content.strip()

        result = json.loads(content)

        # Ensure required fields exist
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
    # Build judge prompt based on context
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

    # Add evaluation criteria based on context
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
        # Generic tool usage without personalization
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

        # Parse JSON response (handle markdown code blocks)
        content = response.content if hasattr(response, "content") else str(response)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        content = content.strip()

        result = json.loads(content)

        # Ensure required fields exist
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
