# Location: app/semantic_index/embedding/verify_setup.py
# Quick verification script to test if all components work

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def verify_components():
    """Verify all components can be imported and initialized"""

    print("=" * 60)
    print("PHASE 3 SETUP VERIFICATION")
    print("=" * 60)

    errors = []

    # 1. Check imports
    print("\n1. Checking imports...")
    try:
        from app.semantic_index.embedding import (
            AccumulatorWriter,
            CoverageMonitor,
            EmbeddingClient,
            EmbeddingWorker,  # noqa: F401
            EnrichmentFetcher,  # noqa: F401
            FingerprintTracker,
            OntologyResolver,
        )

        print("   ✓ All components imported successfully")
    except ImportError as e:
        print(f"   ✗ Import error: {e}")
        errors.append(f"Import error: {e}")
        return errors

    # 2. Check ontology files
    print("\n2. Checking ontology files...")
    ontology_dir = Path("ontology")

    tones_v2 = ontology_dir / "tones_v2.csv"
    genres_v1 = ontology_dir / "genres_v1.csv"

    if tones_v2.exists():
        print(f"   ✓ Found {tones_v2}")
    else:
        print(f"   ✗ Missing {tones_v2}")
        errors.append(f"Missing {tones_v2}")

    if genres_v1.exists():
        print(f"   ✓ Found {genres_v1}")
    else:
        print(f"   ✗ Missing {genres_v1}")
        errors.append(f"Missing {genres_v1}")

    # 3. Test OntologyResolver
    if tones_v2.exists() and genres_v1.exists():
        print("\n3. Testing OntologyResolver...")
        try:
            ontology = OntologyResolver()
            print(f"   ✓ Loaded {len(ontology.tone_id_to_name)} tones")
            print(f"   ✓ Loaded {len(ontology.genre_slug_to_name)} genres")
        except Exception as e:
            print(f"   ✗ OntologyResolver error: {e}")
            errors.append(f"OntologyResolver error: {e}")

    # 4. Test FingerprintTracker
    print("\n4. Testing FingerprintTracker...")
    try:
        tracker = FingerprintTracker(db_path="models/data/enriched_v2/fingerprints.db")
        stats = tracker.get_stats("v2")
        print("   ✓ FingerprintTracker initialized")
        print(f"   ✓ Current embeddings: {stats['total_embedded']}")
    except Exception as e:
        print(f"   ✗ FingerprintTracker error: {e}")
        errors.append(f"FingerprintTracker error: {e}")

    # 5. Test EmbeddingClient
    print("\n5. Testing EmbeddingClient...")
    try:
        embedder = EmbeddingClient(device="cpu")
        print("   ✓ Model loaded")
        print(f"   ✓ Embedding dimension: {embedder.get_embedding_dimension()}")

        # Quick test encoding
        test_text = "Test book — Test author | subjects: fiction | genre: novel"
        embedding = embedder.encode_single(test_text)
        print(f"   ✓ Test encoding successful: shape {embedding.shape}")
    except Exception as e:
        print(f"   ✗ EmbeddingClient error: {e}")
        errors.append(f"EmbeddingClient error: {e}")

    # 6. Test AccumulatorWriter
    print("\n6. Testing AccumulatorWriter...")
    try:
        writer = AccumulatorWriter("models/data/enriched_v2")
        batch_count = writer.get_batch_count()
        print("   ✓ AccumulatorWriter initialized")
        print(f"   ✓ Existing batches: {batch_count}")
    except Exception as e:
        print(f"   ✗ AccumulatorWriter error: {e}")
        errors.append(f"AccumulatorWriter error: {e}")

    # 7. Test CoverageMonitor
    print("\n7. Testing CoverageMonitor...")
    try:
        monitor = CoverageMonitor("models/data/enriched_v2")
        stats = monitor.get_stats()
        print("   ✓ CoverageMonitor initialized")
        print(f"   ✓ Coverage: {stats['coverage_percent']:.2f}%")
    except Exception as e:
        print(f"   ✗ CoverageMonitor error: {e}")
        errors.append(f"CoverageMonitor error: {e}")

    # 8. Check database connection
    print("\n8. Testing database connection...")
    try:
        from app.database import SessionLocal

        db = SessionLocal()
        db.close()
        print("   ✓ Database connection successful")
    except Exception as e:
        print(f"   ✗ Database connection error: {e}")
        errors.append(f"Database connection error: {e}")

    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"VERIFICATION FAILED: {len(errors)} error(s)")
        print("=" * 60)
        for error in errors:
            print(f"  - {error}")
    else:
        print("VERIFICATION PASSED: All components working!")
        print("=" * 60)
        print("\nReady to run:")
        print("  python -m app.semantic_index.embedding.worker --mode full --limit 10")
    print()

    return errors


if __name__ == "__main__":
    errors = verify_components()
    sys.exit(len(errors))
