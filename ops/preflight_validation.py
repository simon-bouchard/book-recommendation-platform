#!/usr/bin/env python3
# ops/preflight_enrichment.py
"""
Pre-flight validation for Phase 2 enrichment pipeline.
Run this before starting any V2 enrichment to catch environment issues early.

Usage:
    python ops/preflight_enrichment.py
    python ops/preflight_enrichment.py --quick  # Skip slow checks
"""
import os
import sys
from pathlib import Path
from typing import Tuple, Optional

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import text
from app.database import SessionLocal
from app.table_models import (
    Tone, Genre, OLSubject, BookOLSubject, 
    Book, Author, EnrichmentError
)

# Colors for terminal output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_header(text: str):
    """Print section header"""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}{text}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")


def print_check(name: str, passed: bool, details: str = ""):
    """Print check result"""
    status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
    print(f"{status} | {name}")
    if details:
        print(f"       {details}")


def check_schema_ontology_version() -> Tuple[bool, str]:
    """Check if ontology_version column exists in tones and genres"""
    try:
        with SessionLocal() as db:
            # Check tones table
            result = db.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA = DATABASE() "
                "AND TABLE_NAME = 'tones' "
                "AND COLUMN_NAME = 'ontology_version'"
            )).scalar()
            
            if result == 0:
                return False, "tones table missing ontology_version column"
            
            # Check genres table
            result = db.execute(text(
                "SELECT COUNT(*) FROM information_schema.COLUMNS "
                "WHERE TABLE_SCHEMA = DATABASE() "
                "AND TABLE_NAME = 'genres' "
                "AND COLUMN_NAME = 'ontology_version'"
            )).scalar()
            
            if result == 0:
                return False, "genres table missing ontology_version column"
            
            return True, "Both tables have ontology_version column"
            
    except Exception as e:
        return False, f"Database error: {str(e)}"


def check_tones_data() -> Tuple[bool, str]:
    """Check if tones v1 and v2 exist with correct counts"""
    try:
        with SessionLocal() as db:
            # Check v1 (flexible - may not be 55 if using custom ontology)
            v1_count = db.query(Tone).filter(
                Tone.ontology_version == 'v1'
            ).count()
            
            if v1_count == 0:
                return False, f"Tones v1: no tones found (need at least some v1 tones)"
            
            # Check v2 (required for Phase 2)
            v2_count = db.query(Tone).filter(
                Tone.ontology_version == 'v2'
            ).count()
            
            if v2_count != 36:
                return False, f"Tones v2: expected 36, found {v2_count} (Phase 2 requires v2)"
            
            # Check ID ranges
            v1_ids = db.query(Tone.tone_id).filter(
                Tone.ontology_version == 'v1'
            ).all()
            v1_min = min(id for (id,) in v1_ids)
            v1_max = max(id for (id,) in v1_ids)
            
            v2_ids = db.query(Tone.tone_id).filter(
                Tone.ontology_version == 'v2'
            ).all()
            v2_min = min(id for (id,) in v2_ids)
            v2_max = max(id for (id,) in v2_ids)
            
            details = (
                f"v1: {v1_count} tones (IDs {v1_min}-{v1_max}), "
                f"v2: {v2_count} tones (IDs {v2_min}-{v2_max})"
            )
            
            return True, details
            
    except Exception as e:
        return False, f"Database error: {str(e)}"


def check_genres_data() -> Tuple[bool, str]:
    """Check if genres v1 exists with correct count"""
    try:
        with SessionLocal() as db:
            v1_count = db.query(Genre).filter(
                Genre.ontology_version == 'v1'
            ).count()
            
            if v1_count != 39:
                return False, f"Genres v1: expected 39, found {v1_count}"
            
            return True, f"Genres v1: {v1_count} genres"
            
    except Exception as e:
        return False, f"Database error: {str(e)}"


def check_ol_subjects() -> Tuple[bool, str]:
    """Check if OL subjects are loaded"""
    try:
        with SessionLocal() as db:
            ol_count = db.query(OLSubject).count()
            
            if ol_count == 0:
                return False, "No OL subjects found - quality classifier will fail"
            
            link_count = db.query(BookOLSubject).count()
            
            if link_count == 0:
                return False, "OL subjects exist but no book links - data incomplete"
            
            # Check coverage
            books_with_ol = db.query(BookOLSubject.item_idx).distinct().count()
            total_books = db.query(Book).count()
            
            coverage_pct = (books_with_ol / total_books * 100) if total_books > 0 else 0
            
            details = (
                f"{ol_count:,} subjects, {link_count:,} book-subject links, "
                f"{coverage_pct:.1f}% book coverage"
            )
            
            return True, details
            
    except Exception as e:
        return False, f"Database error: {str(e)}"


def check_books_data() -> Tuple[bool, str]:
    """Check basic book data integrity"""
    try:
        with SessionLocal() as db:
            total = db.query(Book).count()
            
            if total == 0:
                return False, "No books in database"
            
            # Check books with item_idx (required for enrichment)
            with_idx = db.query(Book).filter(
                Book.item_idx.isnot(None)
            ).count()
            
            if with_idx == 0:
                return False, "No books have item_idx"
            
            # Check books with title and author (critical metadata)
            complete = db.query(Book).filter(
                Book.title.isnot(None),
                Book.title != '',
                Book.author_idx.isnot(None)
            ).count()
            
            incomplete_pct = ((total - complete) / total * 100) if total > 0 else 0
            
            details = (
                f"{total:,} books, {with_idx:,} with item_idx, "
                f"{complete:,} with title+author ({incomplete_pct:.1f}% incomplete)"
            )
            
            return True, details
            
    except Exception as e:
        return False, f"Database error: {str(e)}"


def check_quality_classifier() -> Tuple[bool, str]:
    """Test quality classifier on sample books"""
    try:
        from app.enrichment.quality_classifier import assess_book_quality
        
        # Test cases covering all tiers
        test_cases = [
            # RICH tier
            {
                "title": "Test Rich Book",
                "author": "Author Name",
                "description": " ".join(["detailed description"] * 30),
                "ol_subjects": ["Subject1", "Subject2", "Subject3", "Subject4", "Subject5"],
                "expected_tier": "RICH"
            },
            # SPARSE tier
            {
                "title": "Test Sparse Book",
                "author": "Author Name",
                "description": " ".join(["brief description"] * 15),
                "ol_subjects": ["Subject1", "Subject2", "Subject3"],
                "expected_tier": "SPARSE"
            },
            # MINIMAL tier
            {
                "title": "Test Minimal Book",
                "author": "Author Name",
                "description": "Short description here",
                "ol_subjects": ["Subject1"],
                "expected_tier": "MINIMAL"
            },
            # BASIC tier
            {
                "title": "Test Basic Book",
                "author": "Author Name",
                "description": "",
                "ol_subjects": ["Subject1"],
                "expected_tier": "BASIC"
            },
            # INSUFFICIENT tier
            {
                "title": "",
                "author": "Author Name",
                "description": "Some description",
                "ol_subjects": [],
                "expected_tier": "INSUFFICIENT"
            }
        ]
        
        results = []
        for case in test_cases:
            assessment = assess_book_quality(
                title=case["title"],
                author=case["author"],
                description=case["description"],
                ol_subjects=case["ol_subjects"]
            )
            
            if assessment.tier != case["expected_tier"]:
                return False, (
                    f"Tier mismatch: expected {case['expected_tier']}, "
                    f"got {assessment.tier} (score={assessment.score})"
                )
            
            results.append(f"{assessment.tier}({assessment.score})")
        
        return True, f"All 5 test cases passed: {', '.join(results)}"
        
    except Exception as e:
        return False, f"Classifier error: {str(e)}"


def check_validator() -> Tuple[bool, str]:
    """Test validator with sample payloads"""
    try:
        from app.enrichment.validator import validate_payload
        
        # Get valid IDs from database
        with SessionLocal() as db:
            tone_ids = {t.tone_id for t in db.query(Tone).filter(
                Tone.ontology_version == 'v2'
            ).limit(5).all()}
            
            genre_slugs = {g.slug for g in db.query(Genre).filter(
                Genre.ontology_version == 'v1'
            ).limit(5).all()}
        
        if not tone_ids or not genre_slugs:
            return False, "Cannot test - no ontology data"
        
        # Test RICH tier
        try:
            validate_payload(
                payload={
                    "subjects": ["s1", "s2", "s3", "s4", "s5"],
                    "tone_ids": list(tone_ids)[:2],
                    "genre": list(genre_slugs)[0],
                    "vibe": "A fascinating journey through time and space"
                },
                valid_tone_ids=tone_ids,
                valid_genre_slugs=genre_slugs,
                tier="RICH"
            )
        except ValueError as e:
            return False, f"RICH tier validation failed: {e}"
        
        # Test BASIC tier
        try:
            validate_payload(
                payload={
                    "subjects": [],
                    "tone_ids": [],
                    "genre": list(genre_slugs)[0],
                    "vibe": ""
                },
                valid_tone_ids=tone_ids,
                valid_genre_slugs=genre_slugs,
                tier="BASIC"
            )
        except ValueError as e:
            return False, f"BASIC tier validation failed: {e}"
        
        return True, "RICH and BASIC tier validation passed"
        
    except Exception as e:
        return False, f"Validator error: {str(e)}"


def check_kafka_connectivity() -> Tuple[bool, str]:
    """Check if Kafka is accessible"""
    try:
        from kafka import KafkaProducer
        from kafka.admin import KafkaAdminClient
        
        bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        
        # Try to connect
        admin = KafkaAdminClient(
            bootstrap_servers=bootstrap_servers,
            client_id='preflight-check',
            request_timeout_ms=5000
        )
        
        # Check topics exist
        topics = admin.list_topics()
        
        required_topics = ["enrich.results.v1", "enrich.errors.v1"]
        missing_topics = [t for t in required_topics if t not in topics]
        
        admin.close()
        
        if missing_topics:
            return False, f"Missing topics: {', '.join(missing_topics)}"
        
        return True, f"Connected to {bootstrap_servers}, topics exist"
        
    except ImportError:
        return False, "kafka-python not installed (optional - can skip)"
    except Exception as e:
        return False, f"Kafka connection failed: {str(e)}"


def check_environment_variables() -> Tuple[bool, str]:
    """Check required environment variables"""
    required = {
        "DATABASE_URL": "Database connection string",
        "DEEPINFRA_API_KEY": "LLM API key",
    }
    
    optional = {
        "KAFKA_BOOTSTRAP_SERVERS": "Kafka connection (default: localhost:9092)",
        "ENRICHMENT_JOB_TAG_VERSION": "Tags version (default: v2)",
        "ENRICHMENT_ONTOLOGY_VERSION": "Ontology version (default: v2)",
    }
    
    missing = []
    for var, desc in required.items():
        if not os.getenv(var):
            missing.append(f"{var} ({desc})")
    
    if missing:
        return False, f"Missing required: {', '.join(missing)}"
    
    # Check optional
    optional_missing = []
    for var, desc in optional.items():
        if not os.getenv(var):
            optional_missing.append(var)
    
    details = "All required vars set"
    if optional_missing:
        details += f", missing optional: {', '.join(optional_missing)}"
    
    return True, details


def check_csv_files() -> Tuple[bool, str]:
    """Check if ontology CSV files exist"""
    tones_v2 = ROOT / "ontology" / "tones_v2.csv"
    genres_v1 = ROOT / "ontology" / "genres_v1.csv"
    
    if not tones_v2.exists():
        return False, f"Missing: {tones_v2}"
    
    if not genres_v1.exists():
        return False, f"Missing: {genres_v1}"
    
    # Count lines
    with open(tones_v2) as f:
        tones_lines = sum(1 for _ in f) - 1  # Exclude header
    
    with open(genres_v1) as f:
        genres_lines = sum(1 for _ in f) - 1
    
    details = f"tones_v2.csv: {tones_lines} rows, genres_v1.csv: {genres_lines} rows"
    
    if tones_lines != 36:
        return False, f"tones_v2.csv: expected 36 rows, found {tones_lines}"
    
    if genres_lines != 39:
        return False, f"genres_v1.csv: expected 39 rows, found {genres_lines}"
    
    return True, details


def check_existing_enrichment() -> Tuple[bool, str]:
    """Check if there's existing V2 enrichment data"""
    try:
        with SessionLocal() as db:
            from app.table_models import BookTone
            
            v2_count = db.query(BookTone.item_idx).filter(
                BookTone.tags_version == 'v2'
            ).distinct().count()
            
            if v2_count > 0:
                return True, f"⚠️ Found {v2_count:,} books already enriched with v2"
            
            return True, "No existing v2 data (fresh start)"
            
    except Exception as e:
        return False, f"Database error: {str(e)}"


def run_all_checks(skip_slow: bool = False):
    """Run all pre-flight checks"""
    print(f"\n{BLUE}{'='*70}")
    print(f"PRE-FLIGHT VALIDATION FOR PHASE 2 ENRICHMENT")
    print(f"{'='*70}{RESET}\n")
    
    checks = []
    
    # Section 1: Environment
    print_header("1. ENVIRONMENT CHECKS")
    
    passed, details = check_environment_variables()
    print_check("Environment variables", passed, details)
    checks.append(("Environment", passed))
    
    passed, details = check_csv_files()
    print_check("Ontology CSV files", passed, details)
    checks.append(("CSV Files", passed))
    
    # Section 2: Database Schema
    print_header("2. DATABASE SCHEMA")
    
    passed, details = check_schema_ontology_version()
    print_check("Schema migration (ontology_version)", passed, details)
    checks.append(("Schema", passed))
    
    # Section 3: Ontology Data
    print_header("3. ONTOLOGY DATA")
    
    passed, details = check_tones_data()
    print_check("Tones (v1 + v2)", passed, details)
    checks.append(("Tones", passed))
    
    passed, details = check_genres_data()
    print_check("Genres (v1)", passed, details)
    checks.append(("Genres", passed))
    
    # Section 4: Book Data
    print_header("4. BOOK DATA")
    
    passed, details = check_books_data()
    print_check("Book metadata", passed, details)
    checks.append(("Books", passed))
    
    passed, details = check_ol_subjects()
    print_check("OL subjects", passed, details)
    checks.append(("OL Subjects", passed))
    
    # Section 5: Code Components
    print_header("5. CODE COMPONENTS")
    
    if not skip_slow:
        passed, details = check_quality_classifier()
        print_check("Quality classifier", passed, details)
        checks.append(("Classifier", passed))
        
        passed, details = check_validator()
        print_check("Validator", passed, details)
        checks.append(("Validator", passed))
    else:
        print(f"{YELLOW}⚠ Skipped (--quick mode){RESET}")
    
    # Section 6: Infrastructure (optional)
    print_header("6. INFRASTRUCTURE (OPTIONAL)")
    
    passed, details = check_kafka_connectivity()
    if not passed and "optional" in details.lower():
        print_check("Kafka connectivity", True, details)  # Don't fail on optional
    else:
        print_check("Kafka connectivity", passed, details)
        checks.append(("Kafka", passed))
    
    # Section 7: Existing Data
    print_header("7. EXISTING DATA CHECK")
    
    passed, details = check_existing_enrichment()
    print_check("Existing V2 enrichment", passed, details)
    # This is informational, not a failure
    
    # Summary
    print_header("SUMMARY")
    
    critical_checks = [c for c in checks if c[0] != "Kafka"]  # Kafka is optional
    passed_count = sum(1 for _, passed in critical_checks if passed)
    total_count = len(critical_checks)
    
    if passed_count == total_count:
        print(f"\n{GREEN}✓ ALL CHECKS PASSED ({passed_count}/{total_count}){RESET}")
        print(f"\n{GREEN}Safe to proceed with enrichment!{RESET}")
        print(f"\nNext steps:")
        print(f"  1. Start Spark consumer: docker-compose -f docker/spark-loader/docker-compose.yml up -d")
        print(f"  2. Run enrichment: python -m app.enrichment.runner_kafka --limit 100")
        return 0
    else:
        print(f"\n{RED}✗ FAILED: {total_count - passed_count} checks failed{RESET}")
        print(f"\n{RED}DO NOT proceed with enrichment until issues are resolved.{RESET}")
        print(f"\nFailed checks:")
        for name, passed in checks:
            if not passed:
                print(f"  • {name}")
        return 1


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Pre-flight validation for Phase 2 enrichment"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip slower checks (classifier, validator tests)"
    )
    
    args = parser.parse_args()
    
    try:
        exit_code = run_all_checks(skip_slow=args.quick)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Interrupted by user{RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n{RED}Unexpected error: {e}{RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
