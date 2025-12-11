# evaluation/chatbot/recommendation_agent/validate_queries.py
"""
Query validation script for recommendation agent evaluation.
Verifies that search queries return expected book types before using in evals.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.database import SessionLocal
from app.agents.tools.native_tools import book_semantic_search, subject_hybrid_pool


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

    try:
        # Test 1: Mystery NOT Cozy
        print("\n### Test 1: Mystery NOT Cozy ###\n")

        base_mystery = book_semantic_search(
            query="mystery detective crime investigation thriller", limit=40, db=db
        )

        cozy_mystery = book_semantic_search(
            query="cozy mystery amateur sleuth small town cats bakery tea shop inn", limit=20, db=db
        )

        print(f"Base mysteries: {len(base_mystery.get('books', []))} books")
        print("Sample titles:")
        for book in base_mystery.get("books", [])[:5]:
            print(f"  - {book['title'][:60]}")

        print(f"\nCozy mysteries: {len(cozy_mystery.get('books', []))} books")
        print("Sample titles:")
        for book in cozy_mystery.get("books", [])[:5]:
            print(f"  - {book['title'][:60]}")

        # Check overlap
        base_ids = {b["item_idx"] for b in base_mystery.get("books", [])}
        cozy_ids = {b["item_idx"] for b in cozy_mystery.get("books", [])}
        overlap = len(base_ids & cozy_ids)
        print(f"\nOverlap: {overlap} books (should be < 5)")

        if overlap >= 5:
            print("WARNING: Too much overlap between base and cozy queries!")
        else:
            print("OK: Queries return distinct book sets")

        # Test 2: Thriller NOT Serial Killer
        print("\n### Test 2: Thriller NOT Serial Killer ###\n")

        base_thriller = book_semantic_search(
            query="thriller suspense action espionage political international", limit=40, db=db
        )

        serial_killer = book_semantic_search(
            query="serial killer FBI profiler forensics criminal minds detective", limit=20, db=db
        )

        print(f"Base thrillers: {len(base_thriller.get('books', []))} books")
        print("Sample titles:")
        for book in base_thriller.get("books", [])[:5]:
            print(f"  - {book['title'][:60]}")

        print(f"\nSerial killer books: {len(serial_killer.get('books', []))} books")
        print("Sample titles:")
        for book in serial_killer.get("books", [])[:5]:
            print(f"  - {book['title'][:60]}")

        # Check overlap
        base_ids = {b["item_idx"] for b in base_thriller.get("books", [])}
        serial_ids = {b["item_idx"] for b in serial_killer.get("books", [])}
        overlap = len(base_ids & serial_ids)
        print(f"\nOverlap: {overlap} books (should be < 5)")

        if overlap >= 5:
            print("WARNING: Too much overlap between base and serial killer queries!")
        else:
            print("OK: Queries return distinct book sets")

    finally:
        db.close()


def validate_genre_matching_queries():
    """
    Validate that genre matching queries return expected book types.

    Ensures:
    - Genre queries return actual books in that genre
    - Wrong-genre queries return books in different genres
    - Minimal overlap between correct and wrong genre results
    """
    print("\n" + "=" * 80)
    print("GENRE MATCHING QUERY VALIDATION")
    print("=" * 80)

    db = SessionLocal()

    try:
        # Test 1: Fantasy Genre
        print("\n### Test 1: Fantasy Genre Matching ###\n")

        fantasy_books = subject_hybrid_pool(
            subject_ids=[1378],  # Fantasy
            limit=40,
            db=db,
        )

        wrong_genre = book_semantic_search(
            query="mystery detective crime thriller suspense investigation", limit=20, db=db
        )

        print(f"Fantasy books: {len(fantasy_books.get('books', []))} books")
        print("Sample titles and subjects:")
        for book in fantasy_books.get("books", [])[:5]:
            subjects = book.get("subjects", [])[:3]
            print(f"  - {book['title'][:50]} | Subjects: {subjects}")

        print(f"\nWrong-genre books: {len(wrong_genre.get('books', []))} books")
        print("Sample titles and subjects:")
        for book in wrong_genre.get("books", [])[:5]:
            subjects = book.get("subjects", [])[:3]
            print(f"  - {book['title'][:50]} | Subjects: {subjects}")

        # Check overlap
        fantasy_ids = {b["item_idx"] for b in fantasy_books.get("books", [])}
        wrong_ids = {b["item_idx"] for b in wrong_genre.get("books", [])}
        overlap = len(fantasy_ids & wrong_ids)
        print(f"\nOverlap: {overlap} books (should be < 5)")

        if overlap >= 5:
            print("WARNING: Too much overlap between fantasy and wrong-genre queries!")
        else:
            print("OK: Queries return distinct book sets")

        # Test 2: Historical Fiction Genre
        print("\n### Test 2: Historical Fiction Genre Matching ###\n")

        historical_books = subject_hybrid_pool(
            subject_ids=[1501],  # Historical Fiction
            limit=40,
            db=db,
        )

        wrong_genre = book_semantic_search(
            query="science fiction space fantasy magic futuristic aliens", limit=20, db=db
        )

        print(f"Historical fiction books: {len(historical_books.get('books', []))} books")
        print("Sample titles and subjects:")
        for book in historical_books.get("books", [])[:5]:
            subjects = book.get("subjects", [])[:3]
            print(f"  - {book['title'][:50]} | Subjects: {subjects}")

        print(f"\nWrong-genre books: {len(wrong_genre.get('books', []))} books")
        print("Sample titles and subjects:")
        for book in wrong_genre.get("books", [])[:5]:
            subjects = book.get("subjects", [])[:3]
            print(f"  - {book['title'][:50]} | Subjects: {subjects}")

        # Check overlap
        historical_ids = {b["item_idx"] for b in historical_books.get("books", [])}
        wrong_ids = {b["item_idx"] for b in wrong_genre.get("books", [])}
        overlap = len(historical_ids & wrong_ids)
        print(f"\nOverlap: {overlap} books (should be < 5)")

        if overlap >= 5:
            print("WARNING: Too much overlap between historical and wrong-genre queries!")
        else:
            print("OK: Queries return distinct book sets")

    finally:
        db.close()


def main():
    """
    Run all query validations.
    """
    print("\nVALIDATING EVALUATION QUERIES")
    print("This script verifies that search queries return expected book types.")
    print("Review sample titles to ensure queries are working as intended.\n")

    validate_negative_constraint_queries()
    validate_genre_matching_queries()

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print("\nManually review sample titles above.")
    print("If queries aren't returning expected book types, adjust keywords and re-run.")
    print("\nOnce validated, these queries will be used in the main evaluation suite.")


if __name__ == "__main__":
    main()
