#!/usr/bin/env python3
# evaluation/semantic_search/compare_indexes.py
"""
Multi-index semantic search comparison tool.

Compares multiple semantic search indexes (baseline, baseline_clean, v1_subjects,
v1_full, v2_subjects, v2_full) to evaluate search quality across different
enrichment strategies.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path for imports
FILE_PATH = Path(__file__).resolve().parents[0]
ROOT = FILE_PATH.parents[1]
sys.path.insert(0, str(ROOT))


# Fields to remove from metadata (LLM-generated, subject to hallucinations)
LLM_GENERATED_FIELDS = {
    "subjects",
    "llm_subjects",
    "tone_ids",
    "tones",
    "tone_names",
    "genre",
    "genre_slug",
    "vibe",
    "vibes",
    "tags_version",
    "scores",
    "enrichment_quality",
    "metadata_source",
}


def clean_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove LLM-generated fields from metadata to avoid hallucinations.
    Only keeps original metadata fields like title, author, description, etc.
    """
    return {k: v for k, v in meta.items() if k not in LLM_GENERATED_FIELDS}


def load_semantic_searcher(index_path: str, embedder):
    """Load a SemanticSearcher from the given index directory."""
    # Import here to avoid issues if running outside project
    try:
        from app.semantic_index.search import SemanticSearcher

        return SemanticSearcher(index_path, embedder)
    except ImportError:
        # Fallback: inline implementation
        print("⚠️  Could not import SemanticSearcher, using fallback implementation")
        import faiss
        import numpy as np

        class FallbackSearcher:
            def __init__(self, dir_path: str, embedder):
                self.dir = Path(dir_path)
                self.index = faiss.read_index(str(self.dir / "semantic.faiss"))
                self.ids = np.load(self.dir / "semantic_ids.npy")
                with open(self.dir / "semantic_meta.json", encoding="utf-8") as f:
                    self.meta = json.load(f)
                self.embedder = embedder

            def search(self, query: str, top_k: int = 10):
                qv = self.embedder([query]).astype("float32")
                D, indices = self.index.search(qv, top_k)
                results = []
                for dist, idx in zip(D[0], indices[0]):
                    if idx == -1:
                        continue
                    bid = int(self.ids[idx])
                    # Clean metadata before returning results
                    cleaned_meta = clean_metadata(self.meta[idx])
                    results.append({"book_id": bid, "meta": cleaned_meta})
                return results

        return FallbackSearcher(index_path, embedder)


def load_test_queries(queries_path: Path) -> Dict[str, Any]:
    """Load test queries from JSON file."""
    with open(queries_path, encoding="utf-8") as f:
        return json.load(f)


def fuzzy_match(text1: str, text2: str, threshold: float = 0.8) -> bool:
    """Check if two strings match with fuzzy matching."""
    if not text1 or not text2:
        return False

    text1 = text1.lower().strip()
    text2 = text2.lower().strip()

    # Exact match
    if text1 == text2:
        return True

    # Simple containment check
    if text1 in text2 or text2 in text1:
        return True

    # Levenshtein distance (optional)
    try:
        from difflib import SequenceMatcher

        similarity = SequenceMatcher(None, text1, text2).ratio()
        return similarity >= threshold
    except Exception:
        return False


def check_expected_in_results(
    expected: Dict, results: List[Dict], top_k: Optional[int] = None
) -> Dict[str, Any]:
    """
    Check if expected book appears in results within top K positions.

    Match criteria:
    1. Exact item_idx match (if provided)
    2. Or fuzzy title + author match
    """
    if top_k is None:
        top_k = len(results)

    results_to_check = results[:top_k]

    expected_idx = expected.get("item_idx")
    expected_title = expected.get("title", "")
    expected_author = expected.get("author", "")

    for rank, result in enumerate(results_to_check, 1):
        book_id = result.get("book_id")
        meta = result.get("meta", {})
        result_title = meta.get("title", "")
        result_author = meta.get("author", "")

        # Check item_idx match
        if expected_idx is not None and book_id == expected_idx:
            return {"found": True, "rank": rank, "matched_by": "item_idx"}

        # Check title + author fuzzy match
        title_match = fuzzy_match(expected_title, result_title)
        author_match = fuzzy_match(expected_author, result_author)

        if title_match and author_match:
            return {"found": True, "rank": rank, "matched_by": "title_author"}

    return {"found": False, "rank": None, "matched_by": None}


def run_assertions(
    query: Dict, all_index_results: Dict[str, List[Dict]]
) -> Optional[Dict[str, Any]]:
    """
    Run programmatic assertions for exact_match queries.
    Returns None for non-exact_match queries.
    """
    if query.get("type") != "exact_match":
        return None

    expected_items = query.get("expected_items", [])
    if not expected_items:
        return None

    assertions = {"passed": True, "details": []}

    for expected in expected_items:
        top_k = expected.get("must_appear_in_top", 10)

        detail = {"expected": expected, "results_by_index": {}}

        all_found = True
        for index_name, results in all_index_results.items():
            check = check_expected_in_results(expected, results, top_k)
            detail["results_by_index"][index_name] = {
                "found": check["found"],
                "rank": check["rank"],
                "matched_by": check["matched_by"],
            }
            if not check["found"]:
                all_found = False

        # Assertion passes if book appears in ALL indexes within top K
        if not all_found:
            assertions["passed"] = False

        assertions["details"].append(detail)

    return assertions


def calculate_pairwise_overlap(results1: List[Dict], results2: List[Dict], top_k: int) -> int:
    """Calculate number of overlapping books in top K results between two indexes."""
    ids1 = {r["book_id"] for r in results1[:top_k]}
    ids2 = {r["book_id"] for r in results2[:top_k]}
    return len(ids1 & ids2)


def calculate_author_diversity(results: List[Dict], top_k: int = 10) -> int:
    """Calculate number of unique authors in top K results."""
    authors = {r["meta"].get("author") for r in results[:top_k] if r["meta"].get("author")}
    return len(authors)


def run_multi_index_comparison(
    searchers: Dict[str, Any], queries: List[Dict], top_k: int = 10
) -> Dict[str, Any]:
    """Run comparison across multiple indexes."""
    results = {"queries": [], "index_names": list(searchers.keys())}

    for i, query in enumerate(queries, 1):
        query_text = query.get("text")
        query_type = query.get("type", "quality")

        print(f"\n[{i}/{len(queries)}] Running query: '{query_text}' (type: {query_type})")

        # Run searches on all indexes
        all_results = {}
        for index_name, searcher in searchers.items():
            print(f"  Searching {index_name}...")
            all_results[index_name] = searcher.search(query_text, top_k=top_k)

        # Run assertions for exact_match queries
        assertions = run_assertions(query, all_results)

        if assertions:
            if assertions["passed"]:
                print("  ✅ Assertions PASSED")
            else:
                print("  ❌ Assertions FAILED")

        # Format results for output
        query_result = {
            "query_id": query.get("id"),
            "query_text": query_text,
            "query_type": query_type,
            "complexity_level": query.get("complexity_level", ""),
            "description": query.get("description", ""),
            "manual_review": query.get("manual_review", False),
            "results_by_index": {},
        }

        # Add results for each index
        for index_name, results_list in all_results.items():
            query_result["results_by_index"][index_name] = [
                {
                    "rank": j + 1,
                    "book_id": r["book_id"],
                    "title": r["meta"].get("title"),
                    "author": r["meta"].get("author"),
                }
                for j, r in enumerate(results_list)
            ]

        # Add metrics (pairwise overlaps, diversity, etc.)
        metrics = {}
        index_names = list(searchers.keys())

        # Calculate pairwise overlaps (top 5 and top 10)
        for idx1 in range(len(index_names)):
            for idx2 in range(idx1 + 1, len(index_names)):
                name1 = index_names[idx1]
                name2 = index_names[idx2]
                pair_key = f"{name1}_vs_{name2}"
                metrics[f"overlap_top5_{pair_key}"] = calculate_pairwise_overlap(
                    all_results[name1], all_results[name2], 5
                )
                metrics[f"overlap_top10_{pair_key}"] = calculate_pairwise_overlap(
                    all_results[name1], all_results[name2], 10
                )

        # Calculate author diversity for each index
        for index_name in index_names:
            metrics[f"author_diversity_{index_name}"] = calculate_author_diversity(
                all_results[index_name], top_k=10
            )

        query_result["metrics"] = metrics

        if assertions:
            query_result["assertions"] = assertions

        results["queries"].append(query_result)

    return results


def save_index_results(results: Dict[str, Any], output_dir: Path, timestamp: str):
    """Save separate JSON file for each index's results."""
    for index_name in results["index_names"]:
        index_results = {"index_name": index_name, "queries": []}

        for query in results["queries"]:
            query_data = {
                "query_id": query.get("query_id"),
                "query_text": query["query_text"],
                "query_type": query["query_type"],
                "complexity_level": query.get("complexity_level", ""),
                "description": query.get("description", ""),
                "results": query["results_by_index"].get(index_name, []),
            }

            # Add assertion results if available
            if "assertions" in query and query["query_type"] == "exact_match":
                query_data["assertion_results"] = []
                for detail in query["assertions"]["details"]:
                    result = detail["results_by_index"].get(index_name, {})
                    query_data["assertion_results"].append(
                        {
                            "expected": detail["expected"],
                            "found": result.get("found", False),
                            "rank": result.get("rank"),
                            "matched_by": result.get("matched_by"),
                        }
                    )

            index_results["queries"].append(query_data)

        # Save to file
        output_path = output_dir / f"{index_name}_{timestamp}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(index_results, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {index_name} results to: {output_path}")


def generate_summary_json(results: Dict[str, Any], metadata: Dict[str, Any], output_path: Path):
    """Generate summary JSON with high-level metrics."""
    summary = {
        "metadata": metadata,
        "indexes": results["index_names"],
        "total_queries": len(results["queries"]),
        "exact_match_results": {},
        "query_type_breakdown": {},
    }

    # Count exact match passes per index
    exact_match_queries = [q for q in results["queries"] if q["query_type"] == "exact_match"]

    if exact_match_queries:
        for index_name in results["index_names"]:
            passed = 0
            total = 0

            for query in exact_match_queries:
                if "assertions" in query:
                    total += 1
                    # Check if all expected items found in this index
                    all_found = all(
                        detail["results_by_index"].get(index_name, {}).get("found", False)
                        for detail in query["assertions"]["details"]
                    )
                    if all_found:
                        passed += 1

            summary["exact_match_results"][index_name] = {
                "passed": passed,
                "total": total,
                "pass_rate": passed / total if total > 0 else 0.0,
            }

    # Query type breakdown
    type_counts = {}
    for query in results["queries"]:
        qtype = query["query_type"]
        type_counts[qtype] = type_counts.get(qtype, 0) + 1
    summary["query_type_breakdown"] = type_counts

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✅ Summary saved to: {output_path}")


def generate_html_output(results: Dict[str, Any], metadata: Dict[str, Any], output_path: Path):
    """Generate HTML comparison report showing all indexes side-by-side."""
    index_names = results["index_names"]
    num_indexes = len(index_names)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Multi-Index Semantic Search Comparison</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .header .meta {{
            color: #666;
            font-size: 14px;
        }}
        .summary {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary h2 {{
            margin-top: 0;
            color: #333;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .summary-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #007bff;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            color: #666;
            text-transform: uppercase;
        }}
        .summary-card .value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .section {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .query {{
            margin-bottom: 40px;
            padding-bottom: 30px;
            border-bottom: 1px solid #e0e0e0;
        }}
        .query:last-child {{
            border-bottom: none;
        }}
        .query-header {{
            margin-bottom: 20px;
        }}
        .query-header h3 {{
            margin: 0 0 5px 0;
            color: #333;
        }}
        .query-meta {{
            color: #666;
            font-size: 14px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            margin-left: 10px;
        }}
        .badge.pass {{
            background: #d4edda;
            color: #155724;
        }}
        .badge.fail {{
            background: #f8d7da;
            color: #721c24;
        }}
        .badge.manual {{
            background: #fff3cd;
            color: #856404;
        }}
        .results-grid {{
            display: grid;
            grid-template-columns: repeat({num_indexes}, 1fr);
            gap: 15px;
            margin-top: 20px;
        }}
        .result-column {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
        }}
        .result-column h4 {{
            margin: 0 0 15px 0;
            font-size: 14px;
            color: #666;
            text-transform: uppercase;
            border-bottom: 2px solid #007bff;
            padding-bottom: 8px;
        }}
        .result-item {{
            padding: 10px;
            margin-bottom: 8px;
            background: white;
            border-radius: 4px;
            border-left: 3px solid #007bff;
        }}
        .result-rank {{
            font-weight: bold;
            color: #007bff;
            margin-right: 8px;
        }}
        .result-title {{
            font-weight: 500;
            color: #333;
        }}
        .result-author {{
            font-size: 13px;
            color: #666;
            margin-top: 2px;
        }}
        .metrics {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin: 15px 0;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }}
        .metric {{
            padding: 8px;
        }}
        .metric-label {{
            font-size: 13px;
            color: #666;
        }}
        .metric-value {{
            font-weight: 600;
            color: #333;
            margin-left: 5px;
        }}
        .assertion-details {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin-top: 15px;
        }}
        .assertion-row {{
            display: grid;
            grid-template-columns: 300px repeat({num_indexes}, 1fr);
            gap: 10px;
            margin-bottom: 10px;
            padding: 10px;
            background: white;
            border-radius: 4px;
        }}
        .assertion-expected {{
            font-weight: 500;
        }}
        .assertion-result {{
            text-align: center;
            padding: 4px 8px;
            border-radius: 4px;
        }}
        .assertion-result.found {{
            background: #d4edda;
            color: #155724;
        }}
        .assertion-result.not-found {{
            background: #f8d7da;
            color: #721c24;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Multi-Index Semantic Search Comparison</h1>
        <div class="meta">
            Generated: {metadata["timestamp"]}<br>
            Embedding Model: {metadata["embedding_model"]}<br>
            Top K: {metadata["top_k"]}<br>
            Indexes: {", ".join(index_names)}
        </div>
    </div>

    <div class="summary">
        <h2>Summary</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Total Queries</h3>
                <div class="value">{len(results["queries"])}</div>
            </div>
"""

    # Add exact match pass rates per index
    exact_match_queries = [q for q in results["queries"] if q["query_type"] == "exact_match"]
    if exact_match_queries:
        for index_name in index_names:
            passed = 0
            total = 0
            for query in exact_match_queries:
                if "assertions" in query:
                    total += 1
                    all_found = all(
                        detail["results_by_index"].get(index_name, {}).get("found", False)
                        for detail in query["assertions"]["details"]
                    )
                    if all_found:
                        passed += 1

            pass_rate = (passed / total * 100) if total > 0 else 0
            html += f"""
            <div class="summary-card">
                <h3>{index_name} Pass Rate</h3>
                <div class="value">{passed}/{total} ({pass_rate:.0f}%)</div>
            </div>
"""

    html += """
        </div>
    </div>
"""

    # Exact match queries
    if exact_match_queries:
        html += """
    <div class="section">
        <h2>🎯 Exact Match Queries</h2>
"""

        for query in exact_match_queries:
            passed = query.get("assertions", {}).get("passed", False)
            badge_class = "pass" if passed else "fail"
            badge_text = "PASS" if passed else "FAIL"

            html += f"""
        <div class="query">
            <div class="query-header">
                <h3>"{query["query_text"]}" <span class="badge {badge_class}">{badge_text}</span></h3>
                <div class="query-meta">{query.get("description", "")} | Complexity: {query.get("complexity_level", "N/A")}</div>
            </div>
"""

            # Show assertion details
            if "assertions" in query:
                html += """
            <div class="assertion-details">
                <h4>Expected Items</h4>
"""
                for detail in query["assertions"]["details"]:
                    expected = detail["expected"]
                    html += f"""
                <div class="assertion-row">
                    <div class="assertion-expected">
                        <strong>{expected.get("title", "N/A")}</strong><br>
                        by {expected.get("author", "N/A")}<br>
                        <small>Must appear in top {expected.get("must_appear_in_top", 10)}</small>
                    </div>
"""
                    for index_name in index_names:
                        result = detail["results_by_index"].get(index_name, {})
                        if result.get("found"):
                            html += f"""
                    <div class="assertion-result found">
                        ✓ Rank {result.get("rank", "N/A")}
                    </div>
"""
                        else:
                            html += """
                    <div class="assertion-result not-found">
                        ✗ Not found
                    </div>
"""
                    html += """
                </div>
"""
                html += """
            </div>
"""

            # Show top results
            html += """
            <div class="results-grid">
"""
            for index_name in index_names:
                results_list = query["results_by_index"].get(index_name, [])
                html += f"""
                <div class="result-column">
                    <h4>{index_name}</h4>
"""
                for r in results_list[:5]:
                    html += f"""
                    <div class="result-item">
                        <span class="result-rank">{r["rank"]}.</span>
                        <div>
                            <div class="result-title">{r.get("title") or "Unknown"}</div>
                            <div class="result-author">{r.get("author") or "Unknown"}</div>
                        </div>
                    </div>
"""
                html += """
                </div>
"""
            html += """
            </div>
        </div>
"""

    html += """
    </div>
"""

    # Quality/manual review queries
    quality_queries = [
        q
        for q in results["queries"]
        if q.get("manual_review", False) or q["query_type"] != "exact_match"
    ]

    if quality_queries:
        html += """
    <div class="section">
        <h2>📝 Quality Queries (Manual Review)</h2>
"""

        for query in quality_queries:
            html += f"""
        <div class="query">
            <div class="query-header">
                <h3>"{query["query_text"]}" <span class="badge manual">MANUAL REVIEW</span></h3>
                <div class="query-meta">{query.get("description", "")} | Type: {query["query_type"]} | Complexity: {query.get("complexity_level", "N/A")}</div>
            </div>

            <div class="results-grid">
"""

            for index_name in index_names:
                results_list = query["results_by_index"].get(index_name, [])
                html += f"""
                <div class="result-column">
                    <h4>{index_name}</h4>
"""
                for r in results_list[:8]:
                    html += f"""
                    <div class="result-item">
                        <span class="result-rank">{r["rank"]}.</span>
                        <div>
                            <div class="result-title">{r.get("title") or "Unknown"}</div>
                            <div class="result-author">{r.get("author") or "Unknown"}</div>
                        </div>
                    </div>
"""
                html += """
                </div>
"""

            html += """
            </div>
        </div>
"""

    html += """
    </div>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ HTML report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple semantic search indexes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all indexes
  python compare_indexes.py \\
    --index baseline=~/bookrec/models/data/baseline \\
    --index baseline_clean=~/bookrec/models/data/baseline_clean \\
    --index v1_subjects=~/bookrec/models/data/enriched_v1_subjects \\
    --index v1_full=~/bookrec/models/data/enriched_v1 \\
    --index v2_subjects=~/bookrec/models/data/enriched_v2_subjects \\
    --index v2_full=~/bookrec/models/data/enriched_v2 \\
    --output results/

  # Compare subset of indexes
  python compare_indexes.py \\
    --index baseline=models/data/baseline \\
    --index v1_full=models/data/enriched_v1 \\
    --index v2_full=models/data/enriched_v2
""",
    )

    parser.add_argument(
        "--index",
        action="append",
        dest="indexes",
        required=True,
        metavar="NAME=PATH",
        help="Index to compare (format: name=path). Can be specified multiple times.",
    )
    parser.add_argument(
        "--queries", default=f"{FILE_PATH}/test_queries.json", help="Path to test queries JSON file"
    )
    parser.add_argument(
        "--output", default=f"{FILE_PATH}/results", help="Output directory for results"
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of results to retrieve per query"
    )
    parser.add_argument(
        "--embedder",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model to use",
    )

    args = parser.parse_args()

    # Parse index arguments
    index_paths = {}
    for idx_arg in args.indexes:
        if "=" not in idx_arg:
            print(f"❌ Invalid index format: {idx_arg}")
            print("   Expected format: name=path")
            return 1
        name, path = idx_arg.split("=", 1)
        index_paths[name] = Path(path).expanduser()

    queries_path = Path(args.queries)
    output_dir = Path(args.output)

    print("\n" + "=" * 80)
    print("MULTI-INDEX SEMANTIC SEARCH COMPARISON")
    print("=" * 80)
    print(f"Indexes to compare: {len(index_paths)}")
    for name, path in index_paths.items():
        print(f"  - {name}: {path}")
    print(f"Test queries: {queries_path}")
    print(f"Output dir: {output_dir}")
    print(f"Top K: {args.top_k}")
    print(f"Embedder: {args.embedder}")
    print("=" * 80)

    # Load embedder
    print("\nLoading embedding model...")
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(args.embedder)

        def embedder(texts, **kwargs):
            return model.encode(texts, convert_to_numpy=True, **kwargs)

        print(f"✅ Loaded {args.embedder}")
    except Exception as e:
        print(f"❌ Failed to load embedder: {e}")
        return 1

    # Load indexes
    print("\nLoading indexes...")
    searchers = {}
    for name, path in index_paths.items():
        try:
            searcher = load_semantic_searcher(str(path), embedder)
            searchers[name] = searcher
            print(f"✅ Loaded {name} index ({len(searcher.ids)} books)")
        except Exception as e:
            print(f"❌ Failed to load {name} index: {e}")
            return 1

    # Load test queries
    print("\nLoading test queries...")
    try:
        queries_data = load_test_queries(queries_path)
        queries = queries_data.get("queries", [])
        print(f"✅ Loaded {len(queries)} test queries")
    except Exception as e:
        print(f"❌ Failed to load test queries: {e}")
        return 1

    if not queries:
        print("❌ No queries found in test file")
        return 1

    # Run comparison
    results = run_multi_index_comparison(searchers, queries, top_k=args.top_k)

    # Generate outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "indexes": {name: str(path) for name, path in index_paths.items()},
        "embedding_model": args.embedder,
        "top_k": args.top_k,
        "num_queries": len(queries),
        "test_queries_version": queries_data.get("version", "unknown"),
    }

    print("\nGenerating outputs...")

    # Save individual index results
    save_index_results(results, output_dir, timestamp)

    # Summary JSON
    summary_path = output_dir / f"summary_{timestamp}.json"
    generate_summary_json(results, metadata, summary_path)

    # HTML report
    html_path = output_dir / f"comparison_{timestamp}.html"
    generate_html_output(results, metadata, html_path)

    print("\n" + "=" * 80)
    print("✅ COMPARISON COMPLETE")
    print("=" * 80)
    print("\nView results:")
    print(f"  HTML Report: {html_path}")
    print(f"  Summary: {summary_path}")
    print(f"  Individual index results: {output_dir}/*_{timestamp}.json")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
