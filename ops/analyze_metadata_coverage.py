#!/usr/bin/env python3
# ops/analyze_metadata_coverage.py
"""
Analyze book metadata coverage to inform quality tier design.
Shows distribution of descriptions, OL subjects, and potential tier assignments.
"""
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import statistics

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import func, case
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.table_models import Book, Author, OLSubject, BookOLSubject


def is_valid_description(description: str) -> bool:
    """Check if description is valid (not null, empty, or placeholder)"""
    if not description:
        return False
    
    desc_lower = description.strip().lower()
    
    # Filter out placeholder text
    placeholders = [
        'no description available',
        'no description available.',
        'description not available',
        'n/a',
        'none',
        ''
    ]
    
    return desc_lower not in placeholders


def analyze_description_coverage(db: Session) -> Dict:
    """Analyze description field coverage and length distribution"""
    print("\n" + "="*80)
    print("DESCRIPTION COVERAGE ANALYSIS")
    print("="*80)
    
    # Total books
    total_books = db.query(func.count(Book.item_idx)).scalar()
    
    # Get all descriptions for analysis
    all_descriptions = db.query(Book.description).all()
    
    # Analyze validity
    valid_descriptions = []
    placeholder_count = 0
    null_or_empty = 0
    
    for (desc,) in all_descriptions:
        if not desc or desc.strip() == '':
            null_or_empty += 1
        elif not is_valid_description(desc):
            placeholder_count += 1
        else:
            valid_descriptions.append(desc)
    
    has_valid_desc = len(valid_descriptions)
    
    # Calculate word counts for valid descriptions
    word_counts = []
    length_buckets = defaultdict(int)
    
    for desc in valid_descriptions:
        word_count = len(desc.split())
        word_counts.append(word_count)
        
        # Bucket by length
        if word_count < 10:
            length_buckets['0-9 words'] += 1
        elif word_count < 50:
            length_buckets['10-49 words'] += 1
        elif word_count < 100:
            length_buckets['50-99 words'] += 1
        elif word_count < 200:
            length_buckets['100-199 words'] += 1
        elif word_count < 500:
            length_buckets['200-499 words'] += 1
        else:
            length_buckets['500+ words'] += 1
    
    # Statistics
    print(f"\nTotal books: {total_books:,}")
    print(f"Books with VALID description: {has_valid_desc:,} ({has_valid_desc/total_books*100:.1f}%)")
    print(f"Books with placeholder text: {placeholder_count:,} ({placeholder_count/total_books*100:.1f}%)")
    print(f"Books with null/empty: {null_or_empty:,} ({null_or_empty/total_books*100:.1f}%)")
    print(f"Books without usable description: {total_books - has_valid_desc:,} ({(total_books-has_valid_desc)/total_books*100:.1f}%)")
    
    if word_counts:
        print(f"\nDescription length statistics (valid descriptions only):")
        print(f"  Mean: {statistics.mean(word_counts):.1f} words")
        print(f"  Median: {statistics.median(word_counts):.1f} words")
        print(f"  Min: {min(word_counts)} words")
        print(f"  Max: {max(word_counts)} words")
        print(f"  Std Dev: {statistics.stdev(word_counts):.1f} words")
        
        print(f"\nLength distribution:")
        for bucket in ['0-9 words', '10-49 words', '50-99 words', '100-199 words', '200-499 words', '500+ words']:
            count = length_buckets[bucket]
            pct = count / len(word_counts) * 100
            bar = '█' * int(pct / 2)
            print(f"  {bucket:15} {count:>8,} ({pct:>5.1f}%) {bar}")
    
    return {
        'total': total_books,
        'has_valid_description': has_valid_desc,
        'placeholder_count': placeholder_count,
        'null_or_empty': null_or_empty,
        'no_usable_description': total_books - has_valid_desc,
        'word_counts': word_counts,
        'length_buckets': dict(length_buckets)
    }


def analyze_ol_subjects_coverage(db: Session) -> Dict:
    """Analyze OL subjects coverage and distribution"""
    print("\n" + "="*80)
    print("OL SUBJECTS COVERAGE ANALYSIS")
    print("="*80)
    
    # Total books
    total_books = db.query(func.count(Book.item_idx)).scalar()
    
    # Books with OL subjects
    has_ol_subjects = db.query(
        func.count(func.distinct(BookOLSubject.item_idx))
    ).scalar()
    
    # Subject count distribution
    subject_counts = db.query(
        BookOLSubject.item_idx,
        func.count(BookOLSubject.ol_subject_idx).label('count')
    ).group_by(BookOLSubject.item_idx).all()
    
    count_distribution = defaultdict(int)
    counts_list = []
    
    for item_idx, count in subject_counts:
        counts_list.append(count)
        
        if count == 1:
            count_distribution['1 subject'] += 1
        elif count <= 3:
            count_distribution['2-3 subjects'] += 1
        elif count <= 5:
            count_distribution['4-5 subjects'] += 1
        elif count <= 10:
            count_distribution['6-10 subjects'] += 1
        elif count <= 20:
            count_distribution['11-20 subjects'] += 1
        else:
            count_distribution['20+ subjects'] += 1
    
    print(f"\nTotal books: {total_books:,}")
    print(f"Books with OL subjects: {has_ol_subjects:,} ({has_ol_subjects/total_books*100:.1f}%)")
    print(f"Books without OL subjects: {total_books - has_ol_subjects:,} ({(total_books-has_ol_subjects)/total_books*100:.1f}%)")
    
    if counts_list:
        print(f"\nOL subjects per book statistics:")
        print(f"  Mean: {statistics.mean(counts_list):.1f} subjects")
        print(f"  Median: {statistics.median(counts_list):.1f} subjects")
        print(f"  Min: {min(counts_list)} subjects")
        print(f"  Max: {max(counts_list)} subjects")
        print(f"  Std Dev: {statistics.stdev(counts_list):.1f} subjects")
        
        print(f"\nSubject count distribution:")
        for bucket in ['1 subject', '2-3 subjects', '4-5 subjects', '6-10 subjects', '11-20 subjects', '20+ subjects']:
            count = count_distribution[bucket]
            pct = count / len(counts_list) * 100
            bar = '█' * int(pct / 2)
            print(f"  {bucket:15} {count:>8,} ({pct:>5.1f}%) {bar}")
        
        # Most common OL subjects
        print(f"\nTop 20 most common OL subjects:")
        top_subjects = db.query(
            OLSubject.subject,
            func.count(BookOLSubject.item_idx).label('book_count')
        ).join(BookOLSubject).group_by(OLSubject.subject).order_by(
            func.count(BookOLSubject.item_idx).desc()
        ).limit(20).all()
        
        for i, (subject, count) in enumerate(top_subjects, 1):
            pct = count / total_books * 100
            print(f"  {i:2}. {subject[:60]:60} {count:>8,} ({pct:>5.1f}%)")
    
    return {
        'total': total_books,
        'has_ol_subjects': has_ol_subjects,
        'no_ol_subjects': total_books - has_ol_subjects,
        'subject_counts': counts_list,
        'count_distribution': dict(count_distribution)
    }


def analyze_basic_metadata(db: Session) -> Dict:
    """Analyze title/author coverage"""
    print("\n" + "="*80)
    print("BASIC METADATA (TITLE/AUTHOR) ANALYSIS")
    print("="*80)
    
    total_books = db.query(func.count(Book.item_idx)).scalar()
    
    # Title coverage
    has_title = db.query(func.count(Book.item_idx)).filter(
        Book.title.isnot(None),
        Book.title != ''
    ).scalar()
    
    # Author coverage
    has_author = db.query(func.count(Book.item_idx)).filter(
        Book.author_idx.isnot(None)
    ).scalar()
    
    # Both title and author
    has_both = db.query(func.count(Book.item_idx)).filter(
        Book.title.isnot(None),
        Book.title != '',
        Book.author_idx.isnot(None)
    ).scalar()
    
    print(f"\nTotal books: {total_books:,}")
    print(f"Books with title: {has_title:,} ({has_title/total_books*100:.1f}%)")
    print(f"Books with author: {has_author:,} ({has_author/total_books*100:.1f}%)")
    print(f"Books with both: {has_both:,} ({has_both/total_books*100:.1f}%)")
    print(f"Books missing title or author: {total_books - has_both:,} ({(total_books-has_both)/total_books*100:.1f}%)")
    
    return {
        'total': total_books,
        'has_title': has_title,
        'has_author': has_author,
        'has_both': has_both,
        'missing_critical': total_books - has_both
    }

def _calculate_metadata_score(desc_words: int, ol_subject_count: int) -> tuple[int, str]:
    """
    Score metadata richness on combined signals.
    Returns (score, tier)
    """
    score = 0
    
    # Description contribution (max 60 points)
    if desc_words >= 100:
        score += 60
    elif desc_words >= 50:
        score += 40
    elif desc_words >= 20:
        score += 20
    elif desc_words >= 10:
        score += 10
    
    # OL subjects contribution (max 40 points)
    if ol_subject_count >= 10:
        score += 40
    elif ol_subject_count >= 7:
        score += 30
    elif ol_subject_count >= 5:
        score += 20
    elif ol_subject_count >= 3:
        score += 10
    
    # Assign tier based on combined score
    if score >= 60:
        return score, "RICH"
    elif score >= 30:
        return score, "SPARSE"
    else:
        return score, "MINIMAL"

def simulate_tier_assignment(db: Session, desc_stats: Dict, ol_stats: Dict) -> Dict:
    """
    Simulate tier assignment based on proposed thresholds.
    
    Tiers:
    - RICH: valid description >= 50 words
    - SPARSE: (10 <= valid description < 50 words) OR (no valid description AND has 5+ OL subjects)
    - MINIMAL: Has title/author but doesn't meet RICH or SPARSE criteria
    - INSUFFICIENT: Missing title or author
    """
    print("\n" + "="*80)
    print("TIER ASSIGNMENT SIMULATION")
    print("="*80)
    
    print("\nTier Definitions:")
    print("  RICH:         valid description >= 50 words")
    print("  SPARSE:       (10 <= valid description < 50 words) OR")
    print("                (no valid description AND has 5+ OL subjects)")
    print("  MINIMAL:      Has title/author but doesn't meet RICH/SPARSE")
    print("  INSUFFICIENT: Missing title or author")
    
    # Get all books with their metadata
    books = db.query(
        Book.item_idx,
        Book.title,
        Book.description,
        Book.author_idx,
        func.count(BookOLSubject.ol_subject_idx).label('ol_subject_count')
    ).outerjoin(
        BookOLSubject, Book.item_idx == BookOLSubject.item_idx
    ).group_by(
        Book.item_idx, Book.title, Book.description, Book.author_idx
    ).all()
    
    tier_counts = defaultdict(int)
    tier_examples = defaultdict(list)
    
    for book in books:
        item_idx = book.item_idx
        title = book.title
        description = book.description
        author_idx = book.author_idx
        ol_count = book.ol_subject_count
        
        # Calculate description word count (only if valid)
        desc_words = 0
        if is_valid_description(description):
            desc_words = len(description.split())
        
        # Assign tier
        if not title or not author_idx:
            tier = 'INSUFFICIENT'
        else:
            score, tier = _calculate_metadata_score(desc_words, ol_count)
        
        tier_counts[tier] += 1
        
        # Store examples (up to 5 per tier)
        if len(tier_examples[tier]) < 5:
            tier_examples[tier].append({
                'item_idx': item_idx,
                'title': title[:60] if title else 'N/A',
                'desc_words': desc_words,
                'ol_subjects': ol_count
            })
    
    total = len(books)
    
    print(f"\nTier Distribution (Total: {total:,} books):")
    print("-" * 80)
    
    for tier in ['RICH', 'SPARSE', 'MINIMAL', 'INSUFFICIENT']:
        count = tier_counts[tier]
        pct = count / total * 100 if total > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"  {tier:12} {count:>8,} ({pct:>5.1f}%) {bar}")
    
    # Show examples for each tier
    print("\n" + "="*80)
    print("EXAMPLE BOOKS BY TIER")
    print("="*80)
    
    for tier in ['RICH', 'SPARSE', 'MINIMAL', 'INSUFFICIENT']:
        if tier_examples[tier]:
            print(f"\n{tier} Examples:")
            for ex in tier_examples[tier]:
                print(f"  item_idx={ex['item_idx']:>6} | desc_words={ex['desc_words']:>4} | "
                      f"ol_subjects={ex['ol_subjects']:>3} | {ex['title']}")
    
    return {
        'total': total,
        'tier_counts': dict(tier_counts),
        'tier_examples': dict(tier_examples)
    }


def analyze_coverage_overlap(db: Session) -> Dict:
    """Analyze overlap between valid description and OL subjects"""
    print("\n" + "="*80)
    print("COVERAGE OVERLAP ANALYSIS")
    print("="*80)
    
    # Get all books with their metadata
    all_books = db.query(
        Book.item_idx,
        Book.description,
        func.count(BookOLSubject.ol_subject_idx).label('ol_count')
    ).outerjoin(
        BookOLSubject, Book.item_idx == BookOLSubject.item_idx
    ).group_by(
        Book.item_idx, Book.description
    ).all()
    
    total_books = len(all_books)
    both = 0
    only_desc = 0
    only_ol = 0
    neither = 0
    
    for book in all_books:
        has_valid_desc = is_valid_description(book.description)
        has_ol = book.ol_count > 0
        
        if has_valid_desc and has_ol:
            both += 1
        elif has_valid_desc and not has_ol:
            only_desc += 1
        elif not has_valid_desc and has_ol:
            only_ol += 1
        else:
            neither += 1
    
    print(f"\nOverlap between valid description and OL subjects:")
    print(f"  Both description AND OL subjects:    {both:>8,} ({both/total_books*100:>5.1f}%)")
    print(f"  Only description (no OL subjects):   {only_desc:>8,} ({only_desc/total_books*100:>5.1f}%)")
    print(f"  Only OL subjects (no description):   {only_ol:>8,} ({only_ol/total_books*100:>5.1f}%)")
    print(f"  Neither description nor OL subjects: {neither:>8,} ({neither/total_books*100:>5.1f}%)")
    
    print(f"\nVisualization:")
    print(f"  ┌─────────────────────────────────────┐")
    print(f"  │ Valid Description: {both + only_desc:>6,}       │")
    print(f"  │   ┌─────────────────────────────┐   │")
    print(f"  │   │ Both: {both:>6,}           │   │")
    print(f"  │   └─────────────────────────────┘   │")
    print(f"  │ Only Desc: {only_desc:>6,}           │")
    print(f"  └─────────────────────────────────────┘")
    print(f"  ┌─────────────────────────────────────┐")
    print(f"  │ OL Subjects: {both + only_ol:>6,}            │")
    print(f"  │ Only OL: {only_ol:>6,}                │")
    print(f"  └─────────────────────────────────────┘")
    print(f"  Neither: {neither:>6,}")
    
    return {
        'both': both,
        'only_description': only_desc,
        'only_ol_subjects': only_ol,
        'neither': neither
    }


def main():
    """Run all coverage analyses"""
    print("="*80)
    print("BOOK METADATA COVERAGE ANALYSIS")
    print("Analyzing metadata to inform quality tier design")
    print("="*80)
    
    with SessionLocal() as db:
        # Basic metadata
        basic_stats = analyze_basic_metadata(db)
        
        # Description analysis
        desc_stats = analyze_description_coverage(db)
        
        # OL subjects analysis
        ol_stats = analyze_ol_subjects_coverage(db)
        
        # Overlap analysis
        overlap_stats = analyze_coverage_overlap(db)
        
        # Simulate tier assignment
        tier_stats = simulate_tier_assignment(db, desc_stats, ol_stats)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)
    
    total = basic_stats['total']
    
    print(f"\nTotal books in catalog: {total:,}")
    print(f"\nCritical metadata (title/author): {basic_stats['has_both']:,} ({basic_stats['has_both']/total*100:.1f}%)")
    print(f"  → {basic_stats['missing_critical']:,} books need metadata fixes first")
    
    print(f"\nEnrichment-ready books: {basic_stats['has_both']:,}")
    
    rich_count = tier_stats['tier_counts'].get('RICH', 0)
    sparse_count = tier_stats['tier_counts'].get('SPARSE', 0)
    minimal_count = tier_stats['tier_counts'].get('MINIMAL', 0)
    
    print(f"  RICH tier (valid desc >= 50 words): {rich_count:,} ({rich_count/total*100:.1f}%)")
    print(f"  SPARSE tier: {sparse_count:,} ({sparse_count/total*100:.1f}%)")
    print(f"  MINIMAL tier: {minimal_count:,} ({minimal_count/total*100:.1f}%)")
    
    print(f"\nOL subjects boost:")
    print(f"  Books with OL subjects: {ol_stats['has_ol_subjects']:,} ({ol_stats['has_ol_subjects']/total*100:.1f}%)")
    print(f"  Books with ONLY OL subjects (no valid desc): {overlap_stats['only_ol_subjects']:,}")
    print(f"  → Can upgrade MINIMAL → SPARSE if 5+ OL subjects")
    
    print("\nRecommendations:")
    if rich_count / total > 0.5:
        print("  ✓ Good description coverage - RICH tier strategy will be primary")
    else:
        print("  ⚠ Low description coverage - rely more on OL subjects for SPARSE")
    
    if ol_stats['has_ol_subjects'] / total > 0.7:
        print("  ✓ Excellent OL subject coverage - strong signal for SPARSE tier")
    else:
        print("  ⚠ Lower OL subject coverage - may need more aggressive MINIMAL strategy")
    
    if overlap_stats['only_ol_subjects'] > 0.1 * total:
        print(f"  ✓ {overlap_stats['only_ol_subjects']:,} books rely entirely on OL subjects - "
              "tiering is crucial")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
