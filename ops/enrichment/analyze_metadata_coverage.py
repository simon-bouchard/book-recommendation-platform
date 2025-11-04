#!/usr/bin/env python3
# ops/analyze_metadata_coverage.py
"""
Analyze book metadata coverage to inform quality tier design.
Shows distribution of descriptions, OL subjects, and potential tier assignments.

Updated for 4-tier system: RICH, SPARSE, MINIMAL, BASIC
"""
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import statistics

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
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
    
    4-Tier System:
    - RICH (score >= 60): Full enrichment (5-8 subjects, 2-3 tones, 8-12w vibe)
    - SPARSE (score 30-59): Focused enrichment (3-5 subjects, 0-2 tones, 4-8w vibe optional)
    - MINIMAL (score 10-29): Basic enrichment (1-3 subjects, 0-1 tone, no vibe)
    - BASIC (score < 10): Genre only (0-1 subject, no tones, no vibe)
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
    # < 10 words = 0 points
    
    # OL subjects contribution (max 40 points)
    if ol_subject_count >= 10:
        score += 40
    elif ol_subject_count >= 7:
        score += 30
    elif ol_subject_count >= 5:
        score += 20
    elif ol_subject_count >= 3:
        score += 10
    # < 3 subjects = 0 points
    
    # Assign tier based on combined score
    if score >= 60:
        return score, "RICH"
    elif score >= 30:
        return score, "SPARSE"
    elif score >= 10:
        return score, "MINIMAL"
    else:
        return score, "BASIC"


def simulate_tier_assignment(db: Session, desc_stats: Dict, ol_stats: Dict) -> Dict:
    """
    Simulate tier assignment based on 4-tier combined scoring system.
    
    Tiers (based on combined description + OL subject score):
    - RICH (score >= 60): Rich metadata for full enrichment
    - SPARSE (score 30-59): Moderate metadata, focused enrichment
    - MINIMAL (score 10-29): Limited metadata, conservative enrichment
    - BASIC (score < 10): Very sparse metadata, genre classification only
    - INSUFFICIENT: Missing title or author (cannot enrich)
    """
    print("\n" + "="*80)
    print("TIER ASSIGNMENT SIMULATION (4-TIER SYSTEM)")
    print("="*80)
    
    print("\nScoring System:")
    print("  Description Points (max 60):")
    print("    >= 100 words: 60 pts | 50-99 words: 40 pts | 20-49 words: 20 pts")
    print("    10-19 words: 10 pts  | < 10 words: 0 pts")
    print("\n  OL Subject Points (max 40):")
    print("    >= 10 subjects: 40 pts | 7-9 subjects: 30 pts | 5-6 subjects: 20 pts")
    print("    3-4 subjects: 10 pts   | < 3 subjects: 0 pts")
    
    print("\nTier Thresholds (Combined Score):")
    print("  RICH (>= 60):    Full enrichment (5-8 subjects, 2-3 tones, 8-12w vibe)")
    print("  SPARSE (30-59):  Focused enrichment (3-5 subjects, 0-2 tones, vibe optional)")
    print("  MINIMAL (10-29): Basic enrichment (1-3 subjects, 0-1 tone, no vibe)")
    print("  BASIC (< 10):    Genre only (0-1 subject, no tones, no vibe)")
    print("  INSUFFICIENT:    Missing title/author (skip enrichment)")
    
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
    tier_score_ranges = defaultdict(list)  # Store scores per tier for analysis
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
            score = 0
        else:
            score, tier = _calculate_metadata_score(desc_words, ol_count)
            tier_score_ranges[tier].append(score)
        
        tier_counts[tier] += 1
        
        # Store examples (up to 5 per tier)
        if len(tier_examples[tier]) < 5:
            tier_examples[tier].append({
                'item_idx': item_idx,
                'title': title[:60] if title else 'N/A',
                'desc_words': desc_words,
                'ol_subjects': ol_count,
                'score': score
            })
    
    total = len(books)
    
    print(f"\n{'='*80}")
    print(f"Tier Distribution (Total: {total:,} books)")
    print('='*80)
    
    for tier in ['RICH', 'SPARSE', 'MINIMAL', 'BASIC', 'INSUFFICIENT']:
        count = tier_counts[tier]
        pct = count / total * 100 if total > 0 else 0
        bar = '█' * int(pct / 2)
        
        # Show score range for this tier
        score_info = ""
        if tier in tier_score_ranges and tier_score_ranges[tier]:
            scores = tier_score_ranges[tier]
            score_info = f"  [scores: {min(scores)}-{max(scores)}, avg: {sum(scores)/len(scores):.1f}]"
        
        print(f"  {tier:12} {count:>8,} ({pct:>5.1f}%) {bar}{score_info}")
    
    # Show examples for each tier
    print("\n" + "="*80)
    print("EXAMPLE BOOKS BY TIER")
    print("="*80)
    
    for tier in ['RICH', 'SPARSE', 'MINIMAL', 'BASIC', 'INSUFFICIENT']:
        if tier_examples[tier]:
            print(f"\n{tier} Examples:")
            for ex in tier_examples[tier]:
                print(f"  item_idx={ex['item_idx']:>6} | score={ex.get('score', 0):>3} | "
                      f"desc_words={ex['desc_words']:>4} | ol_subj={ex['ol_subjects']:>3} | "
                      f"{ex['title']}")
    
    return {
        'total': total,
        'tier_counts': dict(tier_counts),
        'tier_score_ranges': dict(tier_score_ranges),
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
    print("BOOK METADATA COVERAGE ANALYSIS (4-TIER SYSTEM)")
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
    print(f"  → {basic_stats['missing_critical']:,} books marked as INSUFFICIENT (skip enrichment)")
    
    print(f"\nEnrichment-ready books: {basic_stats['has_both']:,}")
    
    rich_count = tier_stats['tier_counts'].get('RICH', 0)
    sparse_count = tier_stats['tier_counts'].get('SPARSE', 0)
    minimal_count = tier_stats['tier_counts'].get('MINIMAL', 0)
    basic_count = tier_stats['tier_counts'].get('BASIC', 0)
    
    print(f"  RICH (score >= 60):    {rich_count:>8,} ({rich_count/total*100:>5.1f}%) - Full enrichment")
    print(f"  SPARSE (30-59):        {sparse_count:>8,} ({sparse_count/total*100:>5.1f}%) - Focused enrichment")
    print(f"  MINIMAL (10-29):       {minimal_count:>8,} ({minimal_count/total*100:>5.1f}%) - Basic enrichment")
    print(f"  BASIC (< 10):          {basic_count:>8,} ({basic_count/total*100:>5.1f}%) - Genre only")
    
    print(f"\nMetadata Signal Analysis:")
    print(f"  Books with valid descriptions: {desc_stats['has_valid_description']:,} ({desc_stats['has_valid_description']/total*100:.1f}%)")
    print(f"  Books with OL subjects: {ol_stats['has_ol_subjects']:,} ({ol_stats['has_ol_subjects']/total*100:.1f}%)")
    print(f"  Books with BOTH signals: {overlap_stats['both']:,} ({overlap_stats['both']/total*100:.1f}%)")
    print(f"  Books with ONLY OL subjects: {overlap_stats['only_ol_subjects']:,} ({overlap_stats['only_ol_subjects']/total*100:.1f}%)")
    print(f"  Books with NEITHER signal: {overlap_stats['neither']:,} ({overlap_stats['neither']/total*100:.1f}%)")
    
    print("\nKey Insights:")
    
    # Insight 1: Primary tier distribution
    high_quality = rich_count + sparse_count
    low_quality = minimal_count + basic_count
    
    if high_quality > low_quality:
        print(f"  ✓ {high_quality:,} books ({high_quality/total*100:.1f}%) in RICH/SPARSE tiers")
        print(f"    → Good metadata coverage, tiering will improve quality")
    else:
        print(f"  ⚠ {low_quality:,} books ({low_quality/total*100:.1f}%) in MINIMAL/BASIC tiers")
        print(f"    → Many books have sparse metadata, conservative approach crucial")
    
    # Insight 2: BASIC tier size
    if basic_count > 0.15 * total:
        print(f"  ⚠ BASIC tier is large ({basic_count/total*100:.1f}%)")
        print(f"    → Many books will get genre-only enrichment")
        print(f"    → Consider if threshold adjustment needed after testing")
    elif basic_count > 0.05 * total:
        print(f"  ✓ BASIC tier is reasonable ({basic_count/total*100:.1f}%)")
        print(f"    → Acceptable for very sparse metadata books")
    else:
        print(f"  ✓ BASIC tier is small ({basic_count/total*100:.1f}%)")
        print(f"    → Most books have enough metadata for subject extraction")
    
    # Insight 3: OL subjects boost
    if overlap_stats['only_ol_subjects'] > 0.1 * total:
        print(f"  ✓ {overlap_stats['only_ol_subjects']:,} books rely entirely on OL subjects")
        print(f"    → Combined scoring prevents these from falling to BASIC tier")
    
    # Insight 4: Score distribution
    if 'tier_score_ranges' in tier_stats:
        for tier in ['RICH', 'SPARSE', 'MINIMAL', 'BASIC']:
            if tier in tier_stats['tier_score_ranges'] and tier_stats['tier_score_ranges'][tier]:
                scores = tier_stats['tier_score_ranges'][tier]
                avg_score = sum(scores) / len(scores)
                print(f"  → {tier}: avg score = {avg_score:.1f} (range: {min(scores)}-{max(scores)})")
    
    print("\n" + "="*80)
    print("IMPLEMENTATION READY")
    print("="*80)
    print("\nNext Steps:")
    print("  1. Review tier distribution above")
    print("  2. Verify BASIC threshold (< 10) is appropriate")
    print("  3. Proceed with Phase 2 implementation:")
    print("     - Schema migrations (add ontology_version)")
    print("     - Quality classifier module")
    print("     - Tier-specific prompts")
    print("     - Validator updates")
    print("     - Runner integration")
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
