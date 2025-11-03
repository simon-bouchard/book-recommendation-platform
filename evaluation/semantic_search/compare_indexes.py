#!/usr/bin/env python3
"""
Semantic Search Index Comparison Tool

Compares baseline (raw metadata) vs enriched (LLM-enhanced) semantic search indexes
to evaluate the impact of LLM enrichment on search quality.
"""
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
import argparse
import sys

# Add project root to path for imports
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


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
                D, I = self.index.search(qv, top_k)
                results = []
                for dist, idx in zip(D[0], I[0]):
                    if idx == -1:
                        continue
                    bid = int(self.ids[idx])
                    results.append({
                        "book_id": bid,
                        "score": float(-dist),
                        "meta": self.meta[idx]
                    })
                return results
        
        return FallbackSearcher(index_path, embedder)


def load_test_queries(queries_path: Path) -> Dict[str, Any]:
    """Load test queries from JSON file."""
    with open(queries_path, encoding="utf-8") as f:
        return json.load(f)


def calculate_overlap(baseline_results: List[Dict], enriched_results: List[Dict], top_k: int) -> int:
    """Calculate number of overlapping books in top K results."""
    baseline_ids = {r["book_id"] for r in baseline_results[:top_k]}
    enriched_ids = {r["book_id"] for r in enriched_results[:top_k]}
    return len(baseline_ids & enriched_ids)


def calculate_rank_correlation(baseline_results: List[Dict], enriched_results: List[Dict]) -> float:
    """
    Calculate Spearman's rank correlation between two result lists.
    Returns 0.0 if there's no overlap.
    """
    # Get common book IDs
    baseline_ids = {r["book_id"]: i for i, r in enumerate(baseline_results)}
    enriched_ids = {r["book_id"]: i for i, r in enumerate(enriched_results)}
    common_ids = set(baseline_ids.keys()) & set(enriched_ids.keys())
    
    if len(common_ids) < 2:
        return 0.0
    
    # Get ranks for common books
    baseline_ranks = [baseline_ids[book_id] for book_id in common_ids]
    enriched_ranks = [enriched_ids[book_id] for book_id in common_ids]
    
    # Calculate Spearman's rho
    from scipy.stats import spearmanr
    try:
        corr, _ = spearmanr(baseline_ranks, enriched_ranks)
        return float(corr) if not np.isnan(corr) else 0.0
    except:
        return 0.0


def calculate_metrics(query: Dict, baseline_results: List[Dict], enriched_results: List[Dict]) -> Dict[str, Any]:
    """Calculate comparison metrics for a query."""
    metrics = {}
    
    # Overlap metrics
    metrics["overlap_top_5"] = calculate_overlap(baseline_results, enriched_results, 5)
    metrics["overlap_top_10"] = calculate_overlap(baseline_results, enriched_results, 10)
    
    # Score metrics
    baseline_scores = [r["score"] for r in baseline_results[:10]]
    enriched_scores = [r["score"] for r in enriched_results[:10]]
    
    metrics["baseline_avg_score"] = float(np.mean(baseline_scores)) if baseline_scores else 0.0
    metrics["enriched_avg_score"] = float(np.mean(enriched_scores)) if enriched_scores else 0.0
    
    # Rank correlation
    metrics["rank_correlation"] = calculate_rank_correlation(baseline_results, enriched_results)
    
    # Diversity metrics (optional)
    baseline_authors = {r["meta"].get("author") for r in baseline_results[:10] if r["meta"].get("author")}
    enriched_authors = {r["meta"].get("author") for r in enriched_results[:10] if r["meta"].get("author")}
    
    metrics["baseline_author_diversity"] = len(baseline_authors)
    metrics["enriched_author_diversity"] = len(enriched_authors)
    
    return metrics


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
    except:
        return False


def check_expected_in_results(expected: Dict, results: List[Dict], top_k: Optional[int] = None) -> Dict[str, Any]:
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
            return {
                "found": True,
                "rank": rank,
                "score": result.get("score"),
                "matched_by": "item_idx"
            }
        
        # Check title + author fuzzy match
        title_match = fuzzy_match(expected_title, result_title)
        author_match = fuzzy_match(expected_author, result_author)
        
        if title_match and author_match:
            return {
                "found": True,
                "rank": rank,
                "score": result.get("score"),
                "matched_by": "title_author"
            }
    
    return {
        "found": False,
        "rank": None,
        "score": None,
        "matched_by": None
    }


def run_assertions(query: Dict, baseline_results: List[Dict], enriched_results: List[Dict]) -> Optional[Dict[str, Any]]:
    """
    Run programmatic assertions for exact_match queries.
    Returns None for non-exact_match queries.
    """
    if query.get("type") != "exact_match":
        return None
    
    expected_items = query.get("expected_items", [])
    if not expected_items:
        return None
    
    assertions = {
        "passed": True,
        "details": []
    }
    
    for expected in expected_items:
        must_appear_in_top = expected.get("must_appear_in_top", 10)
        
        # Check baseline
        baseline_check = check_expected_in_results(expected, baseline_results, must_appear_in_top)
        
        # Check enriched
        enriched_check = check_expected_in_results(expected, enriched_results, must_appear_in_top)
        
        # Determine pass/fail
        baseline_pass = baseline_check["found"]
        enriched_pass = enriched_check["found"]
        
        if not baseline_pass or not enriched_pass:
            assertions["passed"] = False
        
        assertions["details"].append({
            "expected_title": expected.get("title"),
            "expected_author": expected.get("author"),
            "expected_item_idx": expected.get("item_idx"),
            "must_appear_in_top": must_appear_in_top,
            "baseline": baseline_check,
            "enriched": enriched_check
        })
    
    return assertions


def format_result_for_output(result: Dict, rank: int, tags_version: str) -> Dict[str, Any]:
    """Format a single search result for output JSON."""
    meta = result.get("meta", {})
    return {
        "rank": rank,
        "item_idx": result.get("book_id"),
        "title": meta.get("title"),
        "author": meta.get("author"),
        "score": result.get("score"),
        "tags_version": tags_version,
        # Include enrichment-specific fields if available
        "subjects": meta.get("subjects"),
        "tone_ids": meta.get("tone_ids"),
        "vibe": meta.get("vibe"),
    }


def run_comparison(
    baseline_searcher,
    enriched_searcher,
    queries: List[Dict],
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Run comparison for all queries.
    Returns list of comparison results.
    """
    results = []
    
    print(f"\nRunning comparison on {len(queries)} queries...")
    print("=" * 80)
    
    for i, query in enumerate(queries, 1):
        query_text = query["text"]
        query_type = query.get("type", "unknown")
        
        print(f"\n[{i}/{len(queries)}] Query: \"{query_text}\" ({query_type})")
        
        # Search both indexes
        try:
            baseline_results = baseline_searcher.search(query_text, top_k=top_k)
        except Exception as e:
            print(f"  ⚠️  Baseline search failed: {e}")
            baseline_results = []
        
        try:
            enriched_results = enriched_searcher.search(query_text, top_k=top_k)
        except Exception as e:
            print(f"  ⚠️  Enriched search failed: {e}")
            enriched_results = []
        
        # Calculate metrics
        metrics = calculate_metrics(query, baseline_results, enriched_results)
        
        # Run assertions for exact_match queries
        assertions = run_assertions(query, baseline_results, enriched_results)
        
        # Print summary
        if assertions:
            passed = assertions["passed"]
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status} - Assertion results")
            for detail in assertions["details"]:
                b_found = detail["baseline"]["found"]
                e_found = detail["enriched"]["found"]
                print(f"    Baseline: {'✓' if b_found else '✗'} (rank {detail['baseline']['rank']})")
                print(f"    Enriched: {'✓' if e_found else '✗'} (rank {detail['enriched']['rank']})")
        else:
            print(f"  Manual review required")
        
        print(f"  Overlap (top 5): {metrics['overlap_top_5']}/5")
        print(f"  Avg scores: Baseline={metrics['baseline_avg_score']:.3f}, Enriched={metrics['enriched_avg_score']:.3f}")
        
        # Format results
        comparison = {
            "query_id": query.get("id"),
            "query_text": query_text,
            "query_type": query_type,
            "description": query.get("description"),
            "manual_review": query.get("manual_review", False),
            
            "baseline_results": [
                format_result_for_output(r, rank, "baseline")
                for rank, r in enumerate(baseline_results, 1)
            ],
            
            "enriched_results": [
                format_result_for_output(r, rank, "enriched_v1")
                for rank, r in enumerate(enriched_results, 1)
            ],
            
            "metrics": metrics,
            "assertions": assertions
        }
        
        results.append(comparison)
    
    print("\n" + "=" * 80)
    print("✅ Comparison complete\n")
    
    return results


def generate_summary(results: List[Dict]) -> Dict[str, Any]:
    """Generate summary statistics from comparison results."""
    total_queries = len(results)
    
    # Count query types
    exact_match_queries = [r for r in results if r["query_type"] == "exact_match"]
    quality_queries = [r for r in results if r.get("manual_review", False)]
    
    # Exact match results
    exact_match_passed = {
        "baseline": 0,
        "enriched": 0,
        "both": 0,
        "neither": 0
    }
    
    for result in exact_match_queries:
        assertions = result.get("assertions")
        if not assertions:
            continue
        
        baseline_pass = all(d["baseline"]["found"] for d in assertions["details"])
        enriched_pass = all(d["enriched"]["found"] for d in assertions["details"])
        
        if baseline_pass:
            exact_match_passed["baseline"] += 1
        if enriched_pass:
            exact_match_passed["enriched"] += 1
        if baseline_pass and enriched_pass:
            exact_match_passed["both"] += 1
        if not baseline_pass and not enriched_pass:
            exact_match_passed["neither"] += 1
    
    # Average scores
    all_baseline_scores = []
    all_enriched_scores = []
    
    for result in results:
        metrics = result.get("metrics", {})
        all_baseline_scores.append(metrics.get("baseline_avg_score", 0.0))
        all_enriched_scores.append(metrics.get("enriched_avg_score", 0.0))
    
    avg_baseline_score = float(np.mean(all_baseline_scores)) if all_baseline_scores else 0.0
    avg_enriched_score = float(np.mean(all_enriched_scores)) if all_enriched_scores else 0.0
    
    # Average overlap
    all_overlap_5 = [r["metrics"].get("overlap_top_5", 0) for r in results]
    all_overlap_10 = [r["metrics"].get("overlap_top_10", 0) for r in results]
    
    summary = {
        "total_queries": total_queries,
        "exact_match_queries": len(exact_match_queries),
        "quality_queries": len(quality_queries),
        
        "exact_match_results": {
            "baseline_passed": exact_match_passed["baseline"],
            "enriched_passed": exact_match_passed["enriched"],
            "both_passed": exact_match_passed["both"],
            "both_failed": exact_match_passed["neither"],
            "baseline_pass_rate": exact_match_passed["baseline"] / len(exact_match_queries) if exact_match_queries else 0.0,
            "enriched_pass_rate": exact_match_passed["enriched"] / len(exact_match_queries) if exact_match_queries else 0.0
        },
        
        "avg_scores": {
            "baseline": avg_baseline_score,
            "enriched": avg_enriched_score,
            "improvement": avg_enriched_score - avg_baseline_score,
            "improvement_pct": ((avg_enriched_score - avg_baseline_score) / avg_baseline_score * 100) if avg_baseline_score > 0 else 0.0
        },
        
        "avg_overlap": {
            "top_5": float(np.mean(all_overlap_5)) if all_overlap_5 else 0.0,
            "top_10": float(np.mean(all_overlap_10)) if all_overlap_10 else 0.0
        }
    }
    
    return summary


def generate_json_output(
    results: List[Dict],
    metadata: Dict[str, Any],
    output_path: Path
) -> None:
    """Generate full comparison JSON output."""
    summary = generate_summary(results)
    
    output = {
        "metadata": metadata,
        "queries": results,
        "summary": summary
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Full JSON saved to: {output_path}")


def generate_summary_json(
    results: List[Dict],
    metadata: Dict[str, Any],
    output_path: Path
) -> None:
    """Generate summary JSON output."""
    summary = generate_summary(results)
    
    # Determine recommendation
    exact_match_better = summary["exact_match_results"]["enriched_pass_rate"] > summary["exact_match_results"]["baseline_pass_rate"]
    score_better = summary["avg_scores"]["improvement"] > 0
    
    if exact_match_better and score_better:
        recommendation = "enriched"
        confidence = "high"
    elif exact_match_better or score_better:
        recommendation = "enriched"
        confidence = "medium"
    else:
        recommendation = "baseline"
        confidence = "low"
    
    output = {
        "comparison_date": metadata["timestamp"],
        "recommendation": recommendation,
        "confidence": confidence,
        "key_metrics": {
            "exact_match_accuracy": {
                "baseline": summary["exact_match_results"]["baseline_pass_rate"],
                "enriched": summary["exact_match_results"]["enriched_pass_rate"],
                "winner": "enriched" if exact_match_better else "baseline"
            },
            "avg_score_improvement": summary["avg_scores"]["improvement"],
            "queries_requiring_manual_review": summary["quality_queries"]
        },
        "programmatic_conclusion": f"{'Enriched' if recommendation == 'enriched' else 'Baseline'} performs better on exact match tests ({summary['exact_match_results']['enriched_pass_rate']:.1%} vs {summary['exact_match_results']['baseline_pass_rate']:.1%} pass rate)",
        "next_steps": [
            f"Review {summary['quality_queries']} quality queries manually or with LLM",
            "Focus on vibe and thematic query performance",
            "Analyze which enrichment features drive improvements"
        ]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Summary JSON saved to: {output_path}")


def generate_html_output(
    results: List[Dict],
    metadata: Dict[str, Any],
    output_path: Path
) -> None:
    """Generate HTML comparison report."""
    summary = generate_summary(results)
    
    # HTML template
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Semantic Search Comparison</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        
        .metadata {{
            color: #7f8c8d;
            font-size: 14px;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #ecf0f1;
        }}
        
        .dashboard {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .stat-card {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 6px;
        }}
        
        .stat-card h3 {{
            font-size: 14px;
            color: #7f8c8d;
            margin-bottom: 8px;
            text-transform: uppercase;
        }}
        
        .stat-card .value {{
            font-size: 28px;
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .stat-card.positive .value {{
            color: #27ae60;
        }}
        
        .stat-card.negative .value {{
            color: #e74c3c;
        }}
        
        .section {{
            margin-bottom: 50px;
        }}
        
        .section h2 {{
            color: #2c3e50;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498db;
        }}
        
        .query {{
            background: #f8f9fa;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 6px;
            border-left: 4px solid #3498db;
        }}
        
        .query.passed {{
            border-left-color: #27ae60;
        }}
        
        .query.failed {{
            border-left-color: #e74c3c;
        }}
        
        .query.manual {{
            border-left-color: #f39c12;
        }}
        
        .query-header {{
            margin-bottom: 15px;
        }}
        
        .query-header h3 {{
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        
        .query-meta {{
            color: #7f8c8d;
            font-size: 14px;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 10px;
        }}
        
        .badge.pass {{
            background: #27ae60;
            color: white;
        }}
        
        .badge.fail {{
            background: #e74c3c;
            color: white;
        }}
        
        .badge.manual {{
            background: #f39c12;
            color: white;
        }}
        
        .results-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 15px;
        }}
        
        .result-column {{
            background: white;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #ddd;
        }}
        
        .result-column h4 {{
            margin-bottom: 10px;
            color: #2c3e50;
        }}
        
        .result-item {{
            padding: 8px 0;
            border-bottom: 1px solid #ecf0f1;
            font-size: 14px;
        }}
        
        .result-item:last-child {{
            border-bottom: none;
        }}
        
        .result-rank {{
            display: inline-block;
            width: 30px;
            font-weight: bold;
            color: #7f8c8d;
        }}
        
        .result-title {{
            font-weight: 500;
            color: #2c3e50;
        }}
        
        .result-author {{
            color: #7f8c8d;
            font-size: 13px;
        }}
        
        .result-score {{
            float: right;
            color: #3498db;
            font-size: 12px;
        }}
        
        .metrics {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 6px;
            margin-top: 15px;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }}
        
        .metric {{
            font-size: 14px;
        }}
        
        .metric-label {{
            color: #7f8c8d;
        }}
        
        .metric-value {{
            font-weight: bold;
            color: #2c3e50;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Semantic Search Comparison</h1>
        <div class="metadata">
            <div><strong>Date:</strong> {metadata['timestamp']}</div>
            <div><strong>Baseline Index:</strong> {metadata['baseline_index']}</div>
            <div><strong>Enriched Index:</strong> {metadata['enriched_index']}</div>
            <div><strong>Embedding Model:</strong> {metadata['embedding_model']}</div>
            <div><strong>Top K:</strong> {metadata['top_k']}</div>
        </div>
        
        <div class="dashboard">
            <div class="stat-card">
                <h3>Total Queries</h3>
                <div class="value">{summary['total_queries']}</div>
            </div>
            <div class="stat-card">
                <h3>Exact Match Pass Rate</h3>
                <div class="value">
                    Baseline: {summary['exact_match_results']['baseline_pass_rate']:.1%}<br>
                    Enriched: {summary['exact_match_results']['enriched_pass_rate']:.1%}
                </div>
            </div>
            <div class="stat-card {'positive' if summary['avg_scores']['improvement'] > 0 else 'negative'}">
                <h3>Avg Score Improvement</h3>
                <div class="value">{summary['avg_scores']['improvement']:+.3f}</div>
                <div>({summary['avg_scores']['improvement_pct']:+.1f}%)</div>
            </div>
            <div class="stat-card">
                <h3>Avg Overlap (Top 10)</h3>
                <div class="value">{summary['avg_overlap']['top_10']:.1f}/10</div>
            </div>
        </div>
"""
    
    # Separate exact match and quality queries
    exact_match_queries = [r for r in results if r["query_type"] == "exact_match"]
    quality_queries = [r for r in results if r.get("manual_review", False)]
    
    # Exact match section
    if exact_match_queries:
        html += """
        <div class="section">
            <h2>📋 Exact Match Tests (Programmatic)</h2>
"""
        
        for result in exact_match_queries:
            assertions = result.get("assertions")
            query_class = "passed" if assertions and assertions.get("passed") else "failed"
            badge_class = "pass" if assertions and assertions.get("passed") else "fail"
            badge_text = "✓ PASS" if assertions and assertions.get("passed") else "✗ FAIL"
            
            html += f"""
            <div class="query {query_class}">
                <div class="query-header">
                    <h3>"{result['query_text']}" <span class="badge {badge_class}">{badge_text}</span></h3>
                    <div class="query-meta">{result.get('description', '')}</div>
                </div>
"""
            
            # Show assertion details
            if assertions and assertions.get("details"):
                for detail in assertions["details"]:
                    baseline_status = "✓" if detail["baseline"]["found"] else "✗"
                    enriched_status = "✓" if detail["enriched"]["found"] else "✗"
                    
                    html += f"""
                <div class="metrics">
                    <div><strong>Expected:</strong> {detail['expected_title']} by {detail['expected_author']}</div>
                    <div>Baseline: {baseline_status} (Rank: {detail['baseline']['rank'] or 'N/A'})</div>
                    <div>Enriched: {enriched_status} (Rank: {detail['enriched']['rank'] or 'N/A'})</div>
                </div>
"""
            
            # Show top results
            html += """
                <div class="results-comparison">
                    <div class="result-column">
                        <h4>Baseline Results</h4>
"""
            
            for r in result["baseline_results"][:5]:
                html += f"""
                        <div class="result-item">
                            <span class="result-rank">{r['rank']}.</span>
                            <span class="result-score">{r['score']:.3f}</span>
                            <div class="result-title">{r['title'] or 'Unknown'}</div>
                            <div class="result-author">{r['author'] or 'Unknown'}</div>
                        </div>
"""
            
            html += """
                    </div>
                    <div class="result-column">
                        <h4>Enriched Results</h4>
"""
            
            for r in result["enriched_results"][:5]:
                html += f"""
                        <div class="result-item">
                            <span class="result-rank">{r['rank']}.</span>
                            <span class="result-score">{r['score']:.3f}</span>
                            <div class="result-title">{r['title'] or 'Unknown'}</div>
                            <div class="result-author">{r['author'] or 'Unknown'}</div>
                        </div>
"""
            
            html += """
                    </div>
                </div>
            </div>
"""
    
    # Quality queries section
    if quality_queries:
        html += """
        <div class="section">
            <h2>🎨 Quality Tests (Manual Review)</h2>
"""
        
        for result in quality_queries:
            html += f"""
            <div class="query manual">
                <div class="query-header">
                    <h3>"{result['query_text']}" <span class="badge manual">MANUAL REVIEW</span></h3>
                    <div class="query-meta">{result.get('description', '')} | Type: {result['query_type']}</div>
                </div>
                
                <div class="metrics">
                    <div class="metrics-grid">
                        <div class="metric">
                            <span class="metric-label">Overlap (top 5):</span>
                            <span class="metric-value">{result['metrics']['overlap_top_5']}/5</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Baseline avg score:</span>
                            <span class="metric-value">{result['metrics']['baseline_avg_score']:.3f}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Enriched avg score:</span>
                            <span class="metric-value">{result['metrics']['enriched_avg_score']:.3f}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Rank correlation:</span>
                            <span class="metric-value">{result['metrics']['rank_correlation']:.2f}</span>
                        </div>
                    </div>
                </div>
                
                <div class="results-comparison">
                    <div class="result-column">
                        <h4>Baseline Results</h4>
"""
            
            for r in result["baseline_results"][:8]:
                html += f"""
                        <div class="result-item">
                            <span class="result-rank">{r['rank']}.</span>
                            <span class="result-score">{r['score']:.3f}</span>
                            <div class="result-title">{r['title'] or 'Unknown'}</div>
                            <div class="result-author">{r['author'] or 'Unknown'}</div>
                        </div>
"""
            
            html += """
                    </div>
                    <div class="result-column">
                        <h4>Enriched Results</h4>
"""
            
            for r in result["enriched_results"][:8]:
                html += f"""
                        <div class="result-item">
                            <span class="result-rank">{r['rank']}.</span>
                            <span class="result-score">{r['score']:.3f}</span>
                            <div class="result-title">{r['title'] or 'Unknown'}</div>
                            <div class="result-author">{r['author'] or 'Unknown'}</div>
                        </div>
"""
            
            html += """
                    </div>
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
        description="Compare baseline vs enriched semantic search indexes"
    )
    parser.add_argument(
        "--baseline",
        default="~/bookrec/models/data/baseline",
        help="Path to baseline index directory"
    )
    parser.add_argument(
        "--enriched",
        default="~/bookrec/models/data/enriched_v1",
        help="Path to enriched index directory"
    )
    parser.add_argument(
        "--queries",
        default="test_queries.json",
        help="Path to test queries JSON file"
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to retrieve per query"
    )
    parser.add_argument(
        "--embedder",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model to use"
    )
    
    args = parser.parse_args()
    
    # Expand paths
    baseline_path = Path(args.baseline).expanduser()
    enriched_path = Path(args.enriched).expanduser()
    queries_path = Path(args.queries)
    output_dir = Path(args.output)
    
    print("\n" + "=" * 80)
    print("SEMANTIC SEARCH INDEX COMPARISON")
    print("=" * 80)
    print(f"Baseline index: {baseline_path}")
    print(f"Enriched index: {enriched_path}")
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
        embedder = lambda texts, **kwargs: model.encode(texts, convert_to_numpy=True, **kwargs)
        print(f"✅ Loaded {args.embedder}")
    except Exception as e:
        print(f"❌ Failed to load embedder: {e}")
        return 1
    
    # Load indexes
    print("\nLoading indexes...")
    try:
        baseline_searcher = load_semantic_searcher(str(baseline_path), embedder)
        print(f"✅ Loaded baseline index ({len(baseline_searcher.ids)} books)")
    except Exception as e:
        print(f"❌ Failed to load baseline index: {e}")
        return 1
    
    try:
        enriched_searcher = load_semantic_searcher(str(enriched_path), embedder)
        print(f"✅ Loaded enriched index ({len(enriched_searcher.ids)} books)")
    except Exception as e:
        print(f"❌ Failed to load enriched index: {e}")
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
    results = run_comparison(
        baseline_searcher,
        enriched_searcher,
        queries,
        top_k=args.top_k
    )
    
    # Generate outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "baseline_index": str(baseline_path),
        "enriched_index": str(enriched_path),
        "embedding_model": args.embedder,
        "top_k": args.top_k,
        "num_queries": len(queries),
        "test_queries_version": queries_data.get("version", "unknown")
    }
    
    print("\nGenerating outputs...")
    
    # Full JSON
    json_path = output_dir / f"comparison_{timestamp}.json"
    generate_json_output(results, metadata, json_path)
    
    # Summary JSON
    summary_path = output_dir / f"summary_{timestamp}.json"
    generate_summary_json(results, metadata, summary_path)
    
    # HTML report
    html_path = output_dir / f"comparison_{timestamp}.html"
    generate_html_output(results, metadata, html_path)
    
    print("\n" + "=" * 80)
    print("✅ COMPARISON COMPLETE")
    print("=" * 80)
    print(f"\nView results:")
    print(f"  HTML Report: {html_path}")
    print(f"  Full JSON: {json_path}")
    print(f"  Summary: {summary_path}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
