#!/usr/bin/env python3
"""
Semantic Search Index Comparison Tool

Compares baseline (raw metadata) vs enriched (LLM-enhanced) semantic search indexes
to evaluate the impact of LLM enrichment on search quality.

MODIFIED: Removes LLM-generated metadata (subjects, tones, genres, vibes) from results
to avoid hallucinations. Only keeps original metadata fields.
"""
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
import argparse
import sys

# Add project root to path for imports
FILE_PATH = Path(__file__).resolve().parents[0]
ROOT = FILE_PATH.parents[1]
print(ROOT)
sys.path.insert(0, str(ROOT))


# Fields to remove from metadata (LLM-generated, subject to hallucinations)
LLM_GENERATED_FIELDS = {
    'subjects',
    'llm_subjects', 
    'tone_ids',
    'tones',
    'genre',
    'genre_slug',
    'vibe',
    'vibes',
    'tags_version',
    'scores',
    'enrichment_quality',
    'metadata_source'
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
                D, I = self.index.search(qv, top_k)
                results = []
                for dist, idx in zip(D[0], I[0]):
                    if idx == -1:
                        continue
                    bid = int(self.ids[idx])
                    # Clean metadata before returning results
                    cleaned_meta = clean_metadata(self.meta[idx])
                    results.append({
                        "book_id": bid,
                        "meta": cleaned_meta
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
    
    # Rank correlation
    metrics["rank_correlation"] = calculate_rank_correlation(baseline_results, enriched_results)
    
    # Diversity metrics (optional) - only using non-LLM fields
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
                "matched_by": "item_idx"
            }
        
        # Check title + author fuzzy match
        title_match = fuzzy_match(expected_title, result_title)
        author_match = fuzzy_match(expected_author, result_author)
        
        if title_match and author_match:
            return {
                "found": True,
                "rank": rank,
                "matched_by": "title_author"
            }
    
    return {
        "found": False,
        "rank": None,
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
        
        expected_title = expected.get("title", "")
        expected_author = expected.get("author", "")
        
        detail = {
            "expected_book": f"{expected_title} by {expected_author}" if expected_title else f"item_idx={expected.get('item_idx')}",
            "must_appear_in_top": must_appear_in_top,
            "baseline_found": baseline_check["found"],
            "baseline_rank": baseline_check["rank"],
            "enriched_found": enriched_check["found"],
            "enriched_rank": enriched_check["rank"]
        }
        
        # Assertion passes if book appears in both indexes within top K
        if not (baseline_check["found"] and enriched_check["found"]):
            assertions["passed"] = False
        
        assertions["details"].append(detail)
    
    return assertions


def run_comparison(baseline_searcher, enriched_searcher, queries: List[Dict], top_k: int = 10) -> Dict[str, Any]:
    """Run comparison between baseline and enriched indexes."""
    results = {
        "exact_match_queries": [],
        "quality_queries": []
    }
    
    exact_match_passed = 0
    exact_match_total = 0
    
    for i, query in enumerate(queries, 1):
        query_text = query.get("text")
        query_type = query.get("type", "quality")
        
        print(f"\n[{i}/{len(queries)}] Running query: '{query_text}' (type: {query_type})")
        
        # Run searches
        baseline_results = baseline_searcher.search(query_text, top_k=top_k)
        enriched_results = enriched_searcher.search(query_text, top_k=top_k)
        
        # Calculate metrics
        metrics = calculate_metrics(query, baseline_results, enriched_results)
        
        # Run assertions for exact_match queries
        assertions = run_assertions(query, baseline_results, enriched_results)
        
        # Format results for output
        result = {
            "query_text": query_text,
            "query_type": query_type,
            "description": query.get("description", ""),
            "metrics": metrics,
            "baseline_results": [
                {
                    "rank": i + 1,
                    "book_id": r["book_id"],
                    "title": r["meta"].get("title"),
                    "author": r["meta"].get("author")
                }
                for i, r in enumerate(baseline_results)
            ],
            "enriched_results": [
                {
                    "rank": i + 1,
                    "book_id": r["book_id"],
                    "title": r["meta"].get("title"),
                    "author": r["meta"].get("author")
                }
                for i, r in enumerate(enriched_results)
            ]
        }
        
        if assertions:
            result["assertions"] = assertions
            exact_match_total += 1
            if assertions["passed"]:
                exact_match_passed += 1
                print(f"  ✅ Assertions PASSED")
            else:
                print(f"  ❌ Assertions FAILED")
                for detail in assertions["details"]:
                    if not (detail["baseline_found"] and detail["enriched_found"]):
                        print(f"     - {detail['expected_book']}: baseline={detail['baseline_rank']}, enriched={detail['enriched_rank']}")
        
        # Categorize by query type
        if query_type == "exact_match":
            results["exact_match_queries"].append(result)
        else:
            results["quality_queries"].append(result)
        
        print(f"  Overlap (top 5): {metrics['overlap_top_5']}/5")
        print(f"  Overlap (top 10): {metrics['overlap_top_10']}/10")
        print(f"  Rank correlation: {metrics['rank_correlation']:.2f}")
    
    # Summary statistics
    results["summary"] = {
        "total_queries": len(queries),
        "exact_match_queries": exact_match_total,
        "exact_match_passed": exact_match_passed,
        "exact_match_failed": exact_match_total - exact_match_passed,
        "quality_queries": len(results["quality_queries"])
    }
    
    return results


def generate_json_output(results: Dict, metadata: Dict, output_path: Path):
    """Generate full JSON output with all results."""
    output = {
        "metadata": metadata,
        "summary": results["summary"],
        "results": {
            "exact_match_queries": results["exact_match_queries"],
            "quality_queries": results["quality_queries"]
        }
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Full JSON saved to: {output_path}")


def generate_summary_json(results: Dict, metadata: Dict, output_path: Path):
    """Generate summary JSON with aggregated metrics."""
    exact_match_queries = results["exact_match_queries"]
    quality_queries = results["quality_queries"]
    
    # Aggregate metrics
    avg_overlap_5 = np.mean([q["metrics"]["overlap_top_5"] for q in quality_queries]) if quality_queries else 0
    avg_overlap_10 = np.mean([q["metrics"]["overlap_top_10"] for q in quality_queries]) if quality_queries else 0
    avg_rank_corr = np.mean([q["metrics"]["rank_correlation"] for q in quality_queries]) if quality_queries else 0
    
    summary = {
        "metadata": metadata,
        "summary_stats": {
            "total_queries": results["summary"]["total_queries"],
            "exact_match_passed": results["summary"]["exact_match_passed"],
            "exact_match_failed": results["summary"]["exact_match_failed"],
            "avg_overlap_top_5": float(avg_overlap_5),
            "avg_overlap_top_10": float(avg_overlap_10),
            "avg_rank_correlation": float(avg_rank_corr)
        },
        "exact_match_details": [
            {
                "query": q["query_text"],
                "passed": q["assertions"]["passed"],
                "details": q["assertions"]["details"]
            }
            for q in exact_match_queries
        ]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Summary JSON saved to: {output_path}")


def generate_html_output(results: Dict, metadata: Dict, output_path: Path):
    """Generate HTML report."""
    exact_match_queries = results["exact_match_queries"]
    quality_queries = results["quality_queries"]
    
    # Calculate summary stats
    avg_overlap_5 = np.mean([q["metrics"]["overlap_top_5"] for q in quality_queries]) if quality_queries else 0
    avg_overlap_10 = np.mean([q["metrics"]["overlap_top_10"] for q in quality_queries]) if quality_queries else 0
    avg_rank_corr = np.mean([q["metrics"]["rank_correlation"] for q in quality_queries]) if quality_queries else 0
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Semantic Search Comparison Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
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
        h1 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .metadata {{
            color: #666;
            font-size: 14px;
            margin-top: 15px;
        }}
        .metadata-item {{
            margin: 5px 0;
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
            gap: 20px;
            margin-top: 20px;
        }}
        .summary-item {{
            text-align: center;
        }}
        .summary-value {{
            font-size: 32px;
            font-weight: bold;
            color: #2196F3;
        }}
        .summary-label {{
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #333;
            border-bottom: 2px solid #2196F3;
            padding-bottom: 10px;
        }}
        .query {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .query.passed {{
            border-left: 4px solid #4CAF50;
        }}
        .query.failed {{
            border-left: 4px solid #f44336;
        }}
        .query.manual {{
            border-left: 4px solid #FF9800;
        }}
        .query-header {{
            margin-bottom: 15px;
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
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .badge.passed {{
            background: #4CAF50;
            color: white;
        }}
        .badge.failed {{
            background: #f44336;
            color: white;
        }}
        .badge.manual {{
            background: #FF9800;
            color: white;
        }}
        .assertions {{
            background: #f9f9f9;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            padding: 15px;
            margin: 15px 0;
        }}
        .assertion-detail {{
            padding: 8px 0;
            border-bottom: 1px solid #e0e0e0;
        }}
        .assertion-detail:last-child {{
            border-bottom: none;
        }}
        .assertion-book {{
            font-weight: 600;
            color: #333;
        }}
        .assertion-result {{
            font-size: 13px;
            color: #666;
            margin-top: 4px;
        }}
        .check {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .cross {{
            color: #f44336;
            font-weight: bold;
        }}
        .metrics {{
            background: #f9f9f9;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            padding: 15px;
            margin: 15px 0;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
        }}
        .metric {{
            display: flex;
            flex-direction: column;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-value {{
            font-size: 20px;
            font-weight: 600;
            color: #333;
            margin-top: 4px;
        }}
        .results-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 15px;
        }}
        .result-column {{
            background: #fafafa;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            padding: 15px;
        }}
        .result-column h4 {{
            margin: 0 0 15px 0;
            color: #333;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .result-item {{
            padding: 10px;
            border-bottom: 1px solid #e0e0e0;
            display: grid;
            grid-template-columns: 30px 1fr;
            gap: 10px;
            align-items: start;
        }}
        .result-item:last-child {{
            border-bottom: none;
        }}
        .result-rank {{
            font-weight: 600;
            color: #666;
        }}
        .result-title {{
            font-weight: 500;
            color: #333;
            font-size: 14px;
        }}
        .result-author {{
            color: #666;
            font-size: 12px;
            margin-top: 2px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Semantic Search Index Comparison Report</h1>
        <div class="metadata">
            <div class="metadata-item"><strong>Generated:</strong> {metadata['timestamp']}</div>
            <div class="metadata-item"><strong>Baseline Index:</strong> {metadata['baseline_index']}</div>
            <div class="metadata-item"><strong>Enriched Index:</strong> {metadata['enriched_index']}</div>
            <div class="metadata-item"><strong>Embedding Model:</strong> {metadata['embedding_model']}</div>
            <div class="metadata-item"><strong>Top K:</strong> {metadata['top_k']}</div>
            <div class="metadata-item"><strong>Test Queries:</strong> {metadata['num_queries']}</div>
        </div>
    </div>
    
    <div class="summary">
        <h2>📊 Summary Statistics</h2>
        <div class="summary-grid">
            <div class="summary-item">
                <div class="summary-value">{results['summary']['exact_match_passed']}/{results['summary']['exact_match_queries']}</div>
                <div class="summary-label">Exact Match Passed</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{avg_overlap_5:.1f}/5</div>
                <div class="summary-label">Avg Overlap (Top 5)</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{avg_overlap_10:.1f}/10</div>
                <div class="summary-label">Avg Overlap (Top 10)</div>
            </div>
            <div class="summary-item">
                <div class="summary-value">{avg_rank_corr:.2f}</div>
                <div class="summary-label">Avg Rank Correlation</div>
            </div>
        </div>
    </div>
"""
    
    # Exact match queries
    if exact_match_queries:
        html += """
    <div class="section">
        <h2>✅ Exact Match Queries (Programmatic Validation)</h2>
"""
        
        for result in exact_match_queries:
            status_class = "passed" if result["assertions"]["passed"] else "failed"
            status_badge = "passed" if result["assertions"]["passed"] else "failed"
            status_text = "PASSED" if result["assertions"]["passed"] else "FAILED"
            
            html += f"""
        <div class="query {status_class}">
            <div class="query-header">
                <h3>"{result['query_text']}" <span class="badge {status_badge}">{status_text}</span></h3>
                <div class="query-meta">{result.get('description', '')} | Type: {result['query_type']}</div>
            </div>
            
            <div class="assertions">
                <strong>Assertions:</strong>
"""
            
            for detail in result["assertions"]["details"]:
                baseline_icon = "✓" if detail["baseline_found"] else "✗"
                enriched_icon = "✓" if detail["enriched_found"] else "✗"
                baseline_class = "check" if detail["baseline_found"] else "cross"
                enriched_class = "check" if detail["enriched_found"] else "cross"
                
                html += f"""
                <div class="assertion-detail">
                    <div class="assertion-book">{detail['expected_book']}</div>
                    <div class="assertion-result">
                        Must appear in top {detail['must_appear_in_top']} | 
                        Baseline: <span class="{baseline_class}">{baseline_icon} {detail['baseline_rank'] or 'Not found'}</span> | 
                        Enriched: <span class="{enriched_class}">{enriched_icon} {detail['enriched_rank'] or 'Not found'}</span>
                    </div>
                </div>
"""
            
            html += """
            </div>
            
            <div class="metrics">
                <div class="metrics-grid">
                    <div class="metric">
                        <span class="metric-label">Overlap (top 5):</span>
                        <span class="metric-value">""" + str(result['metrics']['overlap_top_5']) + """/5</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Overlap (top 10):</span>
                        <span class="metric-value">""" + str(result['metrics']['overlap_top_10']) + """/10</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Rank correlation:</span>
                        <span class="metric-value">""" + f"{result['metrics']['rank_correlation']:.2f}" + """</span>
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
                        <div>
                            <div class="result-title">{r['title'] or 'Unknown'}</div>
                            <div class="result-author">{r['author'] or 'Unknown'}</div>
                        </div>
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
                        <div>
                            <div class="result-title">{r['title'] or 'Unknown'}</div>
                            <div class="result-author">{r['author'] or 'Unknown'}</div>
                        </div>
                    </div>
"""
            
            html += """
                </div>
            </div>
        </div>
"""
    
    # Quality queries
    if quality_queries:
        html += """
    <div class="section">
        <h2>📝 Quality Queries (Manual Review)</h2>
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
                        <span class="metric-label">Overlap (top 10):</span>
                        <span class="metric-value">{result['metrics']['overlap_top_10']}/10</span>
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
                        <div>
                            <div class="result-title">{r['title'] or 'Unknown'}</div>
                            <div class="result-author">{r['author'] or 'Unknown'}</div>
                        </div>
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
                        <div>
                            <div class="result-title">{r['title'] or 'Unknown'}</div>
                            <div class="result-author">{r['author'] or 'Unknown'}</div>
                        </div>
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
        description="Compare baseline vs enriched semantic search indexes (LLM metadata removed)"
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
        default=f"{FILE_PATH}/test_queries.json",
        help="Path to test queries JSON file"
    )
    parser.add_argument(
        "--output",
        default=f"{FILE_PATH}/results",
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
    print("SEMANTIC SEARCH INDEX COMPARISON (NO LLM METADATA)")
    print("=" * 80)
    print(f"Baseline index: {baseline_path}")
    print(f"Enriched index: {enriched_path}")
    print(f"Test queries: {queries_path}")
    print(f"Output dir: {output_dir}")
    print(f"Top K: {args.top_k}")
    print(f"Embedder: {args.embedder}")
    print("\n⚠️  LLM-generated metadata will be removed from results")
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
        "test_queries_version": queries_data.get("version", "unknown"),
        "note": "LLM-generated metadata (subjects, tones, genres, vibes) removed to avoid hallucinations"
    }
    
    print("\nGenerating outputs...")
    
    # Full JSON
    json_path = output_dir / f"comparison_no_llm_{timestamp}.json"
    generate_json_output(results, metadata, json_path)
    
    # Summary JSON
    summary_path = output_dir / f"summary_no_llm_{timestamp}.json"
    generate_summary_json(results, metadata, summary_path)
    
    # HTML report
    html_path = output_dir / f"comparison_no_llm_{timestamp}.html"
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
