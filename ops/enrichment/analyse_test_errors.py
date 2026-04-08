#!/usr/bin/env python3
"""
Analyze enrichment errors directly from SQL.

Usage:
    python ops/enrichment/analyze_enrichment_errors.py --version v2
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.database import SessionLocal
from app.table_models import Author, Book, EnrichmentError


def classify_validation_error(error_msg):
    """Classify validation error into a category."""
    msg = error_msg.lower()

    if "vibe too short" in msg:
        return "VIBE_TOO_SHORT"
    elif "vibe too long" in msg:
        return "VIBE_TOO_LONG"
    elif "near-duplicate" in msg:
        return "NEAR_DUPLICATE_SUBJECTS"
    elif "subject" in msg and "count" in msg:
        if "below" in msg or "minimum" in msg:
            return "SUBJECT_COUNT_TOO_LOW"
        elif "exceed" in msg or "maximum" in msg:
            return "SUBJECT_COUNT_TOO_HIGH"
        else:
            return "SUBJECT_COUNT_WRONG"
    elif "tone" in msg and "count" in msg:
        if "below" in msg or "minimum" in msg:
            return "TONE_COUNT_TOO_LOW"
        elif "exceed" in msg or "maximum" in msg:
            return "TONE_COUNT_TOO_HIGH"
        else:
            return "TONE_COUNT_WRONG"
    elif "invalid genre" in msg:
        return "INVALID_GENRE"
    elif "invalid tone" in msg:
        return "INVALID_TONE_ID"
    else:
        return "OTHER_VALIDATION"


def fetch_errors(tags_version: str):
    """Fetch all errors for a tags version."""
    with SessionLocal() as db:
        error_rows = (
            db.query(EnrichmentError).filter(EnrichmentError.tags_version == tags_version).all()
        )

        errors = []
        for err in error_rows:
            # Try to get book info if not in error
            title = err.title or ""
            author = err.author or ""
            description = ""

            if not title:
                book_result = (
                    db.query(Book, Author)
                    .outerjoin(Author, Book.author_idx == Author.author_idx)
                    .filter(Book.item_idx == err.item_idx)
                    .first()
                )

                if book_result:
                    book, auth = book_result
                    title = book.title or ""
                    author = auth.name if auth else ""
                    description = book.description or ""

            # Parse attempted response
            attempted = {}
            if err.attempted:
                try:
                    attempted = (
                        json.loads(err.attempted)
                        if isinstance(err.attempted, str)
                        else err.attempted
                    )
                except Exception:
                    pass

            errors.append(
                {
                    "item_idx": err.item_idx,
                    "title": title,
                    "author": author,
                    "description": description,
                    "stage": err.stage,
                    "error_code": err.error_code,
                    "error_msg": err.error_msg,
                    "attempted": attempted,
                    "occurrence_count": err.occurrence_count,
                }
            )

        return errors


def main():
    parser = argparse.ArgumentParser(description="Analyze enrichment errors from SQL")
    parser.add_argument("--version", default="v2", help="Tags version (default: v2)")
    args = parser.parse_args()

    print("=" * 80)
    print("ENRICHMENT ERROR ANALYSIS")
    print("=" * 80)
    print(f"Tags version: {args.version}\n")

    print("Fetching errors from SQL...")
    all_errors = fetch_errors(args.version)
    print(f"Found {len(all_errors)} total errors\n")

    if not all_errors:
        print("No errors found!")
        return

    # Filter to validation errors
    validation_errors = [e for e in all_errors if e["stage"] == "validate"]
    other_errors = [e for e in all_errors if e["stage"] != "validate"]

    print("=" * 80)
    print("ERROR BREAKDOWN BY STAGE")
    print("=" * 80)
    print(f"\nValidation errors: {len(validation_errors)}")
    print(f"Other errors: {len(other_errors)}")

    # Classify validation errors
    if validation_errors:
        print("\n" + "=" * 80)
        print("VALIDATION ERROR TYPES")
        print("=" * 80)

        error_types = Counter()
        errors_by_type = defaultdict(list)

        for err in validation_errors:
            error_type = classify_validation_error(err["error_msg"])
            error_types[error_type] += 1
            errors_by_type[error_type].append(err)

        print()
        for error_type, count in error_types.most_common():
            pct = count / len(validation_errors) * 100
            print(f"  {error_type:35s} {count:5d} ({pct:5.1f}%)")

        # Overall tier distribution
        print("\n" + "=" * 80)
        print("TIER DISTRIBUTION (ALL VALIDATION ERRORS)")
        print("=" * 80)

        tier_counts = Counter()
        for err in validation_errors:
            attempted = err.get("attempted", {})
            if attempted and isinstance(attempted, dict):
                tier = attempted.get("tier", "UNKNOWN")
                tier_counts[tier] += 1
            else:
                tier_counts["UNKNOWN"] += 1

        print()
        total_with_tier = sum(tier_counts.values())
        for tier in ["RICH", "SPARSE", "MINIMAL", "BASIC", "UNKNOWN"]:
            count = tier_counts[tier]
            if count > 0:
                pct = (count / total_with_tier * 100) if total_with_tier > 0 else 0
                print(f"  {tier:12s} {count:5d} ({pct:5.1f}%)")

        # Quality score analysis for RICH tier errors
        rich_scores = []
        for err in validation_errors:
            attempted = err.get("attempted", {})
            if attempted and isinstance(attempted, dict):
                if attempted.get("tier") == "RICH":
                    score = attempted.get("score")
                    if score is not None:
                        try:
                            rich_scores.append(float(score))
                        except (ValueError, TypeError):
                            pass  # Skip invalid scores

        if rich_scores:
            print(f"\nRICH tier quality scores (n={len(rich_scores)}):")
            print(f"  Min:    {min(rich_scores):.1f}")
            print(f"  Max:    {max(rich_scores):.1f}")
            print(f"  Mean:   {sum(rich_scores) / len(rich_scores):.1f}")
            print(f"  Median: {sorted(rich_scores)[len(rich_scores) // 2]:.1f}")

            # Score ranges
            score_ranges = {
                "60-65": len([s for s in rich_scores if 60 <= s < 65]),
                "65-70": len([s for s in rich_scores if 65 <= s < 70]),
                "70-80": len([s for s in rich_scores if 70 <= s < 80]),
                "80-90": len([s for s in rich_scores if 80 <= s < 90]),
                "90-100": len([s for s in rich_scores if 90 <= s <= 100]),
            }
            print("\n  Score distribution:")
            for range_name, count in score_ranges.items():
                if count > 0:
                    pct = count / len(rich_scores) * 100
                    print(f"    {range_name}: {count:3d} ({pct:5.1f}%)")

        # Detailed analysis for each type
        for error_type, error_list in sorted(
            errors_by_type.items(), key=lambda x: len(x[1]), reverse=True
        ):
            print("\n" + "=" * 80)
            print(f"{error_type} - {len(error_list)} cases")
            print("=" * 80)

            # Tier distribution for this error type
            type_tier_counts = Counter()
            for err in error_list:
                attempted = err.get("attempted", {})
                if attempted and isinstance(attempted, dict):
                    tier = attempted.get("tier", "UNKNOWN")
                    type_tier_counts[tier] += 1
                else:
                    type_tier_counts["UNKNOWN"] += 1

            print("\nTier breakdown:")
            for tier in ["RICH", "SPARSE", "MINIMAL", "BASIC", "UNKNOWN"]:
                count = type_tier_counts[tier]
                if count > 0:
                    pct = count / len(error_list) * 100
                    print(f"  {tier:12s} {count:5d} ({pct:5.1f}%)")

            num_examples = min(5, len(error_list))
            print(f"\nShowing {num_examples} examples:\n")

            for i, err in enumerate(error_list[:num_examples], 1):
                print(f"Example {i}:")
                print(f"  Item: #{err['item_idx']}")
                print(f"  Title: {err['title'][:70]}")
                print(f"  Error: {err['error_msg'][:120]}")

                attempted = err.get("attempted", {})
                if attempted and isinstance(attempted, dict):
                    tier = attempted.get("tier", "?")
                    raw = attempted.get("raw_response", {})

                    print(f"  Tier: {tier}")

                    if isinstance(raw, dict):
                        if "VIBE" in error_type:
                            vibe = raw.get("vibe", "")
                            word_count = len(vibe.split()) if vibe else 0
                            print(f'  Vibe ({word_count} words): "{vibe}"')

                            if vibe and "-" in vibe:
                                hyphenated = [w for w in vibe.split() if "-" in w]
                                print(f"  ⚠️  Has hyphens: {hyphenated}")

                        elif "SUBJECT" in error_type:
                            subjects = raw.get("subjects", [])
                            print(f"  Subjects ({len(subjects)}): {subjects}")

                        elif "TONE" in error_type:
                            tones = raw.get("tone_ids", [])
                            print(f"  Tone IDs ({len(tones)}): {tones}")

                        elif "GENRE" in error_type:
                            genre = raw.get("genre", "")
                            print(f'  Genre: "{genre}"')

                print()

    # Other errors
    if other_errors:
        print("=" * 80)
        print("OTHER ERRORS (non-validation)")
        print("=" * 80)

        other_breakdown = Counter()
        for err in other_errors:
            key = f"{err['stage']}:{err['error_code']}"
            other_breakdown[key] += 1

        print()
        for error_type, count in other_breakdown.most_common():
            print(f"  {error_type}: {count}")

    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()

    if validation_errors:
        # Hyphen analysis
        vibe_short = errors_by_type.get("VIBE_TOO_SHORT", [])
        if vibe_short:
            hyphen_count = 0
            for e in vibe_short:
                attempted = e.get("attempted", {})
                if attempted and isinstance(attempted, dict):
                    raw = attempted.get("raw_response", {})
                    if isinstance(raw, dict):
                        vibe = raw.get("vibe", "")
                        if vibe and "-" in vibe:
                            hyphen_count += 1

            if hyphen_count > 0:
                print(f"• {hyphen_count}/{len(vibe_short)} short vibe errors contain hyphens")
                print("  → May need to clarify word counting in prompt")

        # Near-duplicates
        if len(errors_by_type.get("NEAR_DUPLICATE_SUBJECTS", [])) > 3:
            count = len(errors_by_type["NEAR_DUPLICATE_SUBJECTS"])
            print(f"• {count} near-duplicate subject errors")
            print("  → Prompt may need more emphasis on subject distinctiveness")

        # Top issue
        if error_types:
            top_error, top_count = error_types.most_common(1)[0]
            pct = top_count / len(validation_errors) * 100
            print(f"\n• Top validation issue: {top_error}")
            print(f"  {top_count} cases ({pct:.1f}% of validation errors)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
