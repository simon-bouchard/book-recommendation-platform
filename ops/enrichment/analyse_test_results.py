#!/usr/bin/env python3
"""
Analyze enrichment results directly from SQL.

Usage:
    python ops/enrichment/analyze_enrichment_results.py --version v2 --limit 100
"""
import sys
import argparse
from pathlib import Path
from collections import Counter, defaultdict

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.database import SessionLocal
from app.table_models import (
    Book, Author, OLSubject, BookOLSubject,
    LLMSubject, BookLLMSubject, BookTone, BookGenre,
    Vibe, BookVibe, EnrichmentError
)


def fetch_enriched_books(tags_version: str, limit: int = 100):
    """Fetch enriched books with all their data."""
    with SessionLocal() as db:
        # Get enriched book IDs
        item_idxs = [
            item_idx for (item_idx,) in
            db.query(BookLLMSubject.item_idx)
            .filter(BookLLMSubject.tags_version == tags_version)
            .distinct()
            .limit(limit)
            .all()
        ]
        
        books = []
        for item_idx in item_idxs:
            # Get book info
            book_result = db.query(Book, Author).outerjoin(
                Author, Book.author_idx == Author.author_idx
            ).filter(Book.item_idx == item_idx).first()
            
            if not book_result:
                continue
            
            book, author = book_result
            
            # Get OL subjects
            ol_subjects = [
                subj for (subj,) in
                db.query(OLSubject.subject)
                .join(BookOLSubject, OLSubject.ol_subject_idx == BookOLSubject.ol_subject_idx)
                .filter(BookOLSubject.item_idx == item_idx)
                .all()
            ]
            
            # Get LLM subjects
            llm_subjects = [
                subj for (subj,) in
                db.query(LLMSubject.subject)
                .join(BookLLMSubject, LLMSubject.llm_subject_idx == BookLLMSubject.llm_subject_idx)
                .filter(
                    BookLLMSubject.item_idx == item_idx,
                    BookLLMSubject.tags_version == tags_version
                )
                .all()
            ]
            
            # Get tones
            tone_ids = [
                tone_id for (tone_id,) in
                db.query(BookTone.tone_id)
                .filter(
                    BookTone.item_idx == item_idx,
                    BookTone.tags_version == tags_version
                )
                .all()
            ]
            
            # Get genre
            genre_result = db.query(BookGenre.genre_slug).filter(
                BookGenre.item_idx == item_idx,
                BookGenre.tags_version == tags_version
            ).first()
            genre = genre_result[0] if genre_result else None
            
            # Get vibe
            vibe_result = db.query(Vibe.text).join(
                BookVibe, Vibe.vibe_id == BookVibe.vibe_id
            ).filter(
                BookVibe.item_idx == item_idx,
                BookVibe.tags_version == tags_version
            ).first()
            vibe = vibe_result[0] if vibe_result else ""
            
            books.append({
                'item_idx': item_idx,
                'title': book.title or "",
                'author': author.name if author else "",
                'description': book.description or "",
                'ol_subjects': ol_subjects,
                'llm_subjects': llm_subjects,
                'tone_ids': tone_ids,
                'genre': genre or "",
                'vibe': vibe
            })
        
        return books


def analyze_results(books):
    """Analyze enrichment results for quality issues."""
    stats = {
        'total': len(books),
        'subject_counts': [],
        'tone_counts': [],
        'vibe_lengths': [],
        'has_duplicates': 0,
        'subject_count_violations': [],
        'tone_count_violations': [],
        'vibe_length_violations': [],
        'genres': Counter()
    }
    
    for book in books:
        # Subject count
        subject_count = len(book['llm_subjects'])
        stats['subject_counts'].append(subject_count)
        
        if subject_count > 8:
            stats['subject_count_violations'].append(book)
        
        # Check for duplicates
        subjects_lower = [s.lower() for s in book['llm_subjects']]
        if len(subjects_lower) != len(set(subjects_lower)):
            stats['has_duplicates'] += 1
            if book not in stats['subject_count_violations']:
                stats['subject_count_violations'].append(book)
        
        # Tone count
        tone_count = len(book['tone_ids'])
        stats['tone_counts'].append(tone_count)
        
        if tone_count > 3:
            stats['tone_count_violations'].append(book)
        
        # Vibe length
        vibe = book['vibe']
        word_count = len(vibe.split()) if vibe else 0
        stats['vibe_lengths'].append(word_count)
        
        # Check vibe violations (assuming RICH tier: 8-12 words)
        if vibe and (word_count < 8 or word_count > 12):
            stats['vibe_length_violations'].append(book)
        
        # Genre
        if book['genre']:
            stats['genres'][book['genre']] += 1
    
    return stats


def analyze_errors(tags_version: str):
    """Analyze errors from SQL."""
    with SessionLocal() as db:
        errors = db.query(EnrichmentError).filter(
            EnrichmentError.tags_version == tags_version
        ).all()
        
        error_breakdown = defaultdict(list)
        for err in errors:
            key = f"{err.stage}:{err.error_code}"
            error_breakdown[key].append({
                'item_idx': err.item_idx,
                'title': err.title or "",
                'error_msg': err.error_msg,
                'attempted': err.attempted
            })
        
        return dict(error_breakdown), len(errors)


def main():
    parser = argparse.ArgumentParser(description="Analyze enrichment results from SQL")
    parser.add_argument("--version", default="v2", help="Tags version (default: v2)")
    parser.add_argument("--limit", type=int, default=100, help="Number of books to analyze")
    args = parser.parse_args()
    
    print("="*80)
    print("ENRICHMENT RESULTS ANALYSIS")
    print("="*80)
    print(f"Tags version: {args.version}")
    print(f"Analyzing up to {args.limit} books\n")
    
    # Fetch enriched books
    print("Fetching enriched books from SQL...")
    books = fetch_enriched_books(args.version, args.limit)
    print(f"Found {len(books)} enriched books\n")
    
    if not books:
        print("No enriched books found!")
        return
    
    # Analyze results
    print("Analyzing results...")
    stats = analyze_results(books)
    
    # Display statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"\nTotal books: {stats['total']}")
    
    # Subject stats
    avg_subjects = sum(stats['subject_counts']) / len(stats['subject_counts'])
    max_subjects = max(stats['subject_counts'])
    print(f"\nSubjects:")
    print(f"  Average: {avg_subjects:.1f}")
    print(f"  Max: {max_subjects}")
    print(f"  Books with >8 subjects: {len(stats['subject_count_violations'])}")
    print(f"  Books with duplicate subjects: {stats['has_duplicates']}")
    
    # Tone stats
    avg_tones = sum(stats['tone_counts']) / len(stats['tone_counts'])
    max_tones = max(stats['tone_counts'])
    print(f"\nTones:")
    print(f"  Average: {avg_tones:.1f}")
    print(f"  Max: {max_tones}")
    print(f"  Books with >3 tones: {len(stats['tone_count_violations'])}")
    
    # Vibe stats
    avg_vibe = sum(stats['vibe_lengths']) / len(stats['vibe_lengths']) if stats['vibe_lengths'] else 0
    print(f"\nVibes:")
    print(f"  Average length: {avg_vibe:.1f} words")
    print(f"  Vibes outside 8-12 words: {len(stats['vibe_length_violations'])}")
    
    # Genre stats
    print(f"\nTop 5 genres:")
    for genre, count in stats['genres'].most_common(5):
        print(f"  {genre}: {count}")
    
    # Show violations
    if stats['subject_count_violations']:
        print("\n" + "="*80)
        print("SUBJECT COUNT VIOLATIONS (>8 or duplicates)")
        print("="*80)
        print("\nShowing first 5 examples:\n")
        
        for i, book in enumerate(stats['subject_count_violations'][:5], 1):
            print(f"Example {i}:")
            print(f"  Item: #{book['item_idx']}")
            print(f"  Title: {book['title'][:70]}")
            print(f"  Subjects ({len(book['llm_subjects'])}): {book['llm_subjects']}")
            
            # Check for duplicates
            subjects_lower = [s.lower() for s in book['llm_subjects']]
            if len(subjects_lower) != len(set(subjects_lower)):
                print(f"  ⚠️  HAS DUPLICATES!")
            
            print()
    
    if stats['vibe_length_violations']:
        print("="*80)
        print("VIBE LENGTH VIOLATIONS (not 8-12 words)")
        print("="*80)
        print("\nShowing first 5 examples:\n")
        
        for i, book in enumerate(stats['vibe_length_violations'][:5], 1):
            vibe = book['vibe']
            word_count = len(vibe.split())
            
            print(f"Example {i}:")
            print(f"  Item: #{book['item_idx']}")
            print(f"  Title: {book['title'][:70]}")
            print(f"  Vibe ({word_count} words): \"{vibe}\"")
            print()
    
    # Analyze errors
    print("="*80)
    print("ERROR ANALYSIS")
    print("="*80)
    
    error_breakdown, total_errors = analyze_errors(args.version)
    print(f"\nTotal errors: {total_errors}")
    
    if error_breakdown:
        print("\nError breakdown:")
        for error_type, error_list in sorted(error_breakdown.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"  {error_type}: {len(error_list)}")
        
        # Show examples of top error
        top_error = max(error_breakdown.items(), key=lambda x: len(x[1]))
        error_type, error_list = top_error
        
        print(f"\n{error_type} - showing 3 examples:")
        for i, err in enumerate(error_list[:3], 1):
            print(f"\n  Example {i}:")
            print(f"    Item: #{err['item_idx']}")
            print(f"    Title: {err['title'][:60]}")
            print(f"    Error: {err['error_msg'][:100]}")
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # Generate insights
    if stats['subject_count_violations']:
        pct = len(stats['subject_count_violations']) / stats['total'] * 100
        print(f"• {len(stats['subject_count_violations'])} books ({pct:.1f}%) violate subject count rules")
        print("  → Validator may not be enforcing max_length=8 properly")
    
    if stats['has_duplicates'] > 0:
        pct = stats['has_duplicates'] / stats['total'] * 100
        print(f"• {stats['has_duplicates']} books ({pct:.1f}%) have duplicate subjects")
        print("  → Validator deduplication not working")
    
    if len(stats['vibe_length_violations']) > stats['total'] * 0.1:
        pct = len(stats['vibe_length_violations']) / stats['total'] * 100
        print(f"• {len(stats['vibe_length_violations'])} books ({pct:.1f}%) have vibes outside 8-12 words")
        print("  → May need to clarify prompt instructions")
    
    if not stats['subject_count_violations'] and not stats['has_duplicates']:
        print("• No major validation issues found!")
        print("  → Results look clean")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
