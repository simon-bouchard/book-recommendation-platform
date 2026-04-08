# evaluation/chatbot/recommendation_agent/validate_queries.py
"""
Query validation script for recommendation agent evaluation.
Verifies that search queries return expected book types before using in evals.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.agents.tools.registry import InternalToolGates, ToolRegistry
from app.database import SessionLocal


def create_test_registry(db):
    """Create a tool registry for testing."""
    gates = InternalToolGates(user_num_ratings=12, warm_threshold=10, profile_allowed=True)

    return ToolRegistry.for_retrieval(
        gates=gates,
        ctx_user=None,  # Don't need user for semantic/subject searches
        ctx_db=db,
    )


def validate_negative_constraint_queries():
    """
    Validate that negative constraint queries return expected book types.

    Ensures:
    - Base queries return main genre books
    - Constraint queries return books matching the constraint
    - Minimal overlap between base and constraint results
    """
    print("=" * 80)
    print("NEGATIVE CONSTRAINT QUERY VALIDATION")
    print("=" * 80)

    db = SessionLocal()
    registry = create_test_registry(db)
    semantic_tool = registry.get_tool("book_semantic_search")

    if not semantic_tool:
        print("ERROR: book_semantic_search tool not available")
        return

    try:
        # Test 1: Mystery NOT Cozy
        print("\n### Test 1: Mystery NOT Cozy ###\n")

        base_mystery = semantic_tool.execute(
            query="mystery detective crime investigation thriller", top_k=40
        )

        cozy_mystery = semantic_tool.execute(
            query="cozy mystery amateur sleuth small town cats bakery tea shop inn", top_k=20
        )

        print(f"Base mysteries: {len(base_mystery)} books")
        print("Sample titles:")
        for book in base_mystery[:5]:
            print(f"  - {book.get('title', 'N/A')[:60]}")

        print(f"\nCozy mysteries: {len(cozy_mystery)} books")
        print("Sample titles:")
        for book in cozy_mystery[:5]:
            print(f"  - {book.get('title', 'N/A')[:60]}")

        # Check overlap
        base_ids = {b["item_idx"] for b in base_mystery}
        cozy_ids = {b["item_idx"] for b in cozy_mystery}
        overlap = len(base_ids & cozy_ids)
        print(f"\nOverlap: {overlap} books (should be < 5)")

        if overlap >= 5:
            print("⚠️  WARNING: Too much overlap between base and cozy queries!")
        else:
            print("✅ OK: Queries return distinct book sets")

        # Test 2: Thriller NOT Serial Killer
        print("\n### Test 2: Thriller NOT Serial Killer ###\n")

        base_thriller = semantic_tool.execute(
            query="thriller suspense action espionage political international", top_k=40
        )

        serial_killer = semantic_tool.execute(
            query="serial killer FBI profiler forensics criminal minds detective", top_k=20
        )

        print(f"Base thrillers: {len(base_thriller)} books")
        print("Sample titles:")
        for book in base_thriller[:5]:
            print(f"  - {book.get('title', 'N/A')[:60]}")

        print(f"\nSerial killer books: {len(serial_killer)} books")
        print("Sample titles:")
        for book in serial_killer[:5]:
            print(f"  - {book.get('title', 'N/A')[:60]}")

        # Check overlap
        base_ids = {b["item_idx"] for b in base_thriller}
        serial_ids = {b["item_idx"] for b in serial_killer}
        overlap = len(base_ids & serial_ids)
        print(f"\nOverlap: {overlap} books (should be < 5)")

        if overlap >= 5:
            print("⚠️  WARNING: Too much overlap between base and serial killer queries!")
        else:
            print("✅ OK: Queries return distinct book sets")

    finally:
        db.close()


def validate_genre_queries():
    """
    Validate that genre queries return expected genre books.

    Ensures:
    - Fantasy query returns fantasy books
    - Historical fiction query returns historical books
    - Wrong-genre books are clearly distinct
    """
    print("\n" + "=" * 80)
    print("GENRE QUERY VALIDATION")
    print("=" * 80)

    db = SessionLocal()
    registry = create_test_registry(db)

    semantic_tool = registry.get_tool("book_semantic_search")
    subject_tool = registry.get_tool("subject_hybrid_pool")

    if not semantic_tool or not subject_tool:
        print("ERROR: Required tools not available")
        return

    try:
        # Test 1: Fantasy Genre
        print("\n### Test 1: Fantasy Genre ###\n")

        fantasy_books = subject_tool.execute(
            top_k=40,
            fav_subjects_idxs=[1378],  # Fantasy subject
            weight=0.7,
        )

        wrong_genre = semantic_tool.execute(
            query="mystery detective crime thriller suspense", top_k=20
        )

        print(f"Fantasy books (from subject): {len(fantasy_books)} books")
        print("Sample titles:")
        for book in fantasy_books[:5]:
            print(f"  - {book.get('title', 'N/A')[:60]}")

        print(f"\nWrong genre (mystery/thriller): {len(wrong_genre)} books")
        print("Sample titles:")
        for book in wrong_genre[:5]:
            print(f"  - {book.get('title', 'N/A')[:60]}")

        # Check overlap
        fantasy_ids = {b["item_idx"] for b in fantasy_books}
        wrong_ids = {b["item_idx"] for b in wrong_genre}
        overlap = len(fantasy_ids & wrong_ids)
        print(f"\nOverlap: {overlap} books (should be 0)")

        if overlap > 0:
            print("⚠️  WARNING: Fantasy and mystery queries have overlap!")
        else:
            print("✅ OK: Genres are completely distinct")

        # Test 2: Historical Fiction
        print("\n### Test 2: Historical Fiction ###\n")

        historical_books = subject_tool.execute(
            top_k=40,
            fav_subjects_idxs=[1198],  # Historical Fiction subject
            weight=0.7,
        )

        wrong_genre = semantic_tool.execute(
            query="science fiction fantasy space alien dragon magic", top_k=20
        )

        print(f"Historical fiction books: {len(historical_books)} books")
        print("Sample titles:")
        for book in historical_books[:5]:
            print(f"  - {book.get('title', 'N/A')[:60]}")

        print(f"\nWrong genre (sci-fi/fantasy): {len(wrong_genre)} books")
        print("Sample titles:")
        for book in wrong_genre[:5]:
            print(f"  - {book.get('title', 'N/A')[:60]}")

        # Check overlap
        historical_ids = {b["item_idx"] for b in historical_books}
        wrong_ids = {b["item_idx"] for b in wrong_genre}
        overlap = len(historical_ids & wrong_ids)
        print(f"\nOverlap: {overlap} books (should be 0)")

        if overlap > 0:
            print("⚠️  WARNING: Historical and sci-fi/fantasy queries have overlap!")
        else:
            print("✅ OK: Genres are completely distinct")

    finally:
        db.close()


def main():
    """Run all query validations."""
    print("\n📊 Query Validation for Recommendation Agent Evaluation")
    print("=" * 80)
    print("This script validates that evaluation queries return expected book types.\n")

    # Validate negative constraint queries
    validate_negative_constraint_queries()

    # Validate genre queries
    validate_genre_queries()

    print("\n" + "=" * 80)
    print("✅ Query validation complete!")
    print("=" * 80)
    print("\nIf all queries passed validation, you can run the full evaluation suite.")
    print("If any warnings appeared, consider adjusting the query keywords.\n")


if __name__ == "__main__":
    main()
