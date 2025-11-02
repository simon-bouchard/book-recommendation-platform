#!/usr/bin/env python3
"""
Analyze enrichment results directly from SQL and generate HTML report.

Usage:
    python ops/enrichment/analyze_enrichment_results.py --version v2 --limit 100
    
Generates:
    - enrichment_report_v2_TIMESTAMP.html
"""
import sys
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import html

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
                .distinct()
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


def generate_html_report(books, stats, error_breakdown, total_errors, tags_version):
    """Generate an interactive HTML report for viewing enrichment results."""
    
    # Calculate averages
    avg_subjects = sum(stats['subject_counts']) / len(stats['subject_counts']) if stats['subject_counts'] else 0
    max_subjects = max(stats['subject_counts']) if stats['subject_counts'] else 0
    avg_tones = sum(stats['tone_counts']) / len(stats['tone_counts']) if stats['tone_counts'] else 0
    max_tones = max(stats['tone_counts']) if stats['tone_counts'] else 0
    avg_vibe = sum(stats['vibe_lengths']) / len(stats['vibe_lengths']) if stats['vibe_lengths'] else 0
    
    # Get set of item_idxs that have errors
    error_item_idxs = set()
    for error_list in error_breakdown.values():
        for err in error_list:
            error_item_idxs.add(err['item_idx'])
    
    # Classify books by metadata quality
    for book in books:
        desc_len = len(book['description'])
        if desc_len == 0:
            book['metadata_quality'] = 'none'
        elif desc_len < 100:
            book['metadata_quality'] = 'poor'
        elif desc_len < 500:
            book['metadata_quality'] = 'medium'
        else:
            book['metadata_quality'] = 'rich'
        
        # Add validation flags
        subjects_lower = [s.lower() for s in book['llm_subjects']]
        book['has_duplicates'] = len(subjects_lower) != len(set(subjects_lower))
        book['subject_count_violation'] = len(book['llm_subjects']) > 8
        vibe_words = len(book['vibe'].split()) if book['vibe'] else 0
        book['vibe_violation'] = vibe_words > 0 and (vibe_words < 8 or vibe_words > 12)
        
        # Mark if book has errors
        book['has_error'] = book['item_idx'] in error_item_idxs
    
    # Generate JSON data for JavaScript
    import json
    books_json = json.dumps(books, ensure_ascii=False)
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Enrichment Report - {tags_version}</title>
    <style>
        * {{
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }}
        
        /* Stats Section */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-card h3 {{
            margin: 0 0 8px 0;
            color: #3498db;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .stat-value {{
            font-size: 28px;
            font-weight: bold;
            color: #2c3e50;
            margin: 5px 0;
        }}
        .stat-label {{
            color: #7f8c8d;
            font-size: 12px;
        }}
        
        /* Filter Controls */
        .filters {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .filter-row {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
            margin-bottom: 15px;
        }}
        .filter-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .filter-group label {{
            font-weight: 600;
            color: #555;
            font-size: 14px;
        }}
        select, input[type="text"] {{
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }}
        .filter-stats {{
            color: #7f8c8d;
            font-size: 14px;
            margin-top: 10px;
        }}
        button {{
            padding: 8px 16px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}
        button:hover {{
            background: #2980b9;
        }}
        
        /* Book Cards */
        .book-card {{
            background: white;
            padding: 20px;
            margin: 15px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #3498db;
        }}
        .book-card.violation {{
            border-left-color: #e74c3c;
        }}
        .book-card.poor-metadata {{
            border-left-color: #f39c12;
        }}
        .book-title {{
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        .book-author {{
            color: #7f8c8d;
            font-size: 14px;
            margin-bottom: 15px;
        }}
        .book-description {{
            color: #555;
            font-size: 14px;
            line-height: 1.6;
            margin: 15px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 4px;
        }}
        .subjects-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 15px;
        }}
        .subjects-box {{
            padding: 12px;
            border-radius: 4px;
        }}
        .subjects-box h4 {{
            margin: 0 0 10px 0;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .ol-subjects {{
            background: #ecf0f1;
            border: 1px solid #bdc3c7;
        }}
        .ol-subjects h4 {{
            color: #7f8c8d;
        }}
        .llm-subjects {{
            background: #e8f4f8;
            border: 1px solid #3498db;
        }}
        .llm-subjects h4 {{
            color: #3498db;
        }}
        .tag {{
            display: inline-block;
            background: white;
            padding: 4px 10px;
            margin: 3px;
            border-radius: 3px;
            font-size: 13px;
            border: 1px solid #ddd;
        }}
        .llm-subjects .tag {{
            background: white;
            border-color: #3498db;
            color: #2c3e50;
        }}
        .metadata {{
            display: flex;
            gap: 15px;
            margin-top: 15px;
            flex-wrap: wrap;
        }}
        .metadata-item {{
            background: #f8f9fa;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 13px;
        }}
        .metadata-item strong {{
            color: #3498db;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 3px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .badge.none {{ background: #e74c3c; color: white; }}
        .badge.poor {{ background: #f39c12; color: white; }}
        .badge.medium {{ background: #f39c12; color: white; }}
        .badge.rich {{ background: #27ae60; color: white; }}
        .badge.error {{ background: #e74c3c; color: white; margin-left: 8px; }}
        .vibe {{
            background: #fff9e6;
            border: 1px solid #f39c12;
            padding: 12px;
            border-radius: 4px;
            margin-top: 12px;
            font-style: italic;
            color: #555;
        }}
        .warning {{
            background: #fee;
            border: 2px solid #e74c3c;
            color: #c0392b;
            padding: 8px;
            border-radius: 4px;
            margin-top: 10px;
            font-weight: bold;
            font-size: 13px;
        }}
        .hidden {{
            display: none;
        }}
    </style>
</head>
<body>
    <h1>📚 Enrichment Analysis Report</h1>
    <p><strong>Version:</strong> {html.escape(tags_version)} | <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>📊 Statistics Overview</h2>
    <div class="stats-grid">
        <div class="stat-card">
            <h3>Total Books</h3>
            <div class="stat-value">{stats['total']}</div>
            <div class="stat-label">Enriched books</div>
        </div>
        <div class="stat-card">
            <h3>Subjects</h3>
            <div class="stat-value">{avg_subjects:.1f}</div>
            <div class="stat-label">Avg per book (max: {max_subjects})</div>
        </div>
        <div class="stat-card">
            <h3>Tones</h3>
            <div class="stat-value">{avg_tones:.1f}</div>
            <div class="stat-label">Avg per book (max: {max_tones})</div>
        </div>
        <div class="stat-card">
            <h3>Vibe Length</h3>
            <div class="stat-value">{avg_vibe:.1f}</div>
            <div class="stat-label">Avg words (target: 8-12)</div>
        </div>
        <div class="stat-card">
            <h3>Violations</h3>
            <div class="stat-value">{len(stats['subject_count_violations'])}</div>
            <div class="stat-label">Subject/duplicate issues</div>
        </div>
        <div class="stat-card">
            <h3>Errors</h3>
            <div class="stat-value">{total_errors}</div>
            <div class="stat-label">Processing failures</div>
        </div>
    </div>
    
    <h2>🔍 Browse & Filter Results</h2>
    <div class="filters">
        <div class="filter-row">
            <div class="filter-group">
                <label>Metadata Quality:</label>
                <select id="qualityFilter">
                    <option value="all">All</option>
                    <option value="rich">Rich (>500 chars)</option>
                    <option value="medium">Medium (100-500 chars)</option>
                    <option value="poor">Poor (<100 chars)</option>
                    <option value="none">None (no description)</option>
                </select>
            </div>
            
            <div class="filter-group">
                <label>Validation:</label>
                <select id="validationFilter">
                    <option value="all">All</option>
                    <option value="clean">Clean (no issues)</option>
                    <option value="violations">Has violations</option>
                    <option value="duplicates">Has duplicates</option>
                    <option value="too-many">Too many subjects (>8)</option>
                    <option value="vibe-wrong">Vibe length wrong</option>
                </select>
            </div>
            
            <div class="filter-group">
                <label>Errors:</label>
                <select id="errorFilter">
                    <option value="all">All</option>
                    <option value="no-errors">No errors (success only)</option>
                    <option value="has-errors">Has errors</option>
                </select>
            </div>
            
            <div class="filter-group">
                <label>Genre:</label>
                <select id="genreFilter">
                    <option value="all">All</option>
"""
    
    # Add genre options
    for genre, _ in stats['genres'].most_common(20):
        html_content += f"""                    <option value="{html.escape(genre)}">{html.escape(genre)}</option>\n"""
    
    html_content += f"""
                </select>
            </div>
            
            <div class="filter-group">
                <label>Search:</label>
                <input type="text" id="searchBox" placeholder="Title or author...">
            </div>
            
            <button onclick="resetFilters()">Reset Filters</button>
        </div>
        
        <div class="filter-stats">
            Showing <strong id="visibleCount">{len(books)}</strong> of <strong>{len(books)}</strong> books
        </div>
    </div>
    
    <div id="bookContainer">
        <!-- Books will be inserted here by JavaScript -->
    </div>
    
    <script>
        const allBooks = {books_json};
        
        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}
        
        function renderBooks(books) {{
            const container = document.getElementById('bookContainer');
            container.innerHTML = '';
            
            books.forEach((book, index) => {{
                const hasViolation = book.has_duplicates || book.subject_count_violation;
                const cardClass = hasViolation ? 'book-card violation' : 
                                 book.metadata_quality === 'poor' || book.metadata_quality === 'none' ? 'book-card poor-metadata' : 
                                 'book-card';
                
                const vibeWords = book.vibe ? book.vibe.split(' ').length : 0;
                
                let html = `
                    <div class="${{cardClass}}" data-index="${{index}}">
                        <div class="book-title">${{escapeHtml(book.title)}}</div>
                        <div class="book-author">
                            by ${{escapeHtml(book.author)}} | Item #${{book.item_idx}}
                            <span class="badge ${{book.metadata_quality}}">${{book.metadata_quality}} metadata</span>
                            ${{book.has_error ? '<span class="badge error">HAS ERROR</span>' : ''}}
                        </div>
                `;
                
                if (book.description) {{
                    const desc = book.description.length > 300 ? 
                                book.description.substring(0, 300) + '...' : 
                                book.description;
                    html += `<div class="book-description">${{escapeHtml(desc)}}</div>`;
                }}
                
                html += `
                        <div class="subjects-comparison">
                            <div class="subjects-box ol-subjects">
                                <h4>Original OL Subjects (${{book.ol_subjects.length}})</h4>
                `;
                
                if (book.ol_subjects.length > 0) {{
                    book.ol_subjects.slice(0, 10).forEach(subj => {{
                        html += `<span class="tag">${{escapeHtml(subj)}}</span>`;
                    }});
                }} else {{
                    html += `<span class="tag">None</span>`;
                }}
                
                html += `
                            </div>
                            <div class="subjects-box llm-subjects">
                                <h4>🤖 LLM Enriched Subjects (${{book.llm_subjects.length}})</h4>
                `;
                
                book.llm_subjects.forEach(subj => {{
                    html += `<span class="tag">${{escapeHtml(subj)}}</span>`;
                }});
                
                html += `
                            </div>
                        </div>
                        <div class="metadata">
                            <div class="metadata-item"><strong>Subject Count:</strong> ${{book.llm_subjects.length}}</div>
                `;
                
                if (book.genre) {{
                    html += `<div class="metadata-item"><strong>Genre:</strong> ${{escapeHtml(book.genre)}}</div>`;
                }}
                
                if (book.tone_ids.length > 0) {{
                    html += `<div class="metadata-item"><strong>Tones:</strong> ${{book.tone_ids.join(', ')}}</div>`;
                }}
                
                html += `</div>`;
                
                if (book.vibe) {{
                    html += `<div class="vibe"><strong>Vibe (${{vibeWords}} words):</strong> "${{escapeHtml(book.vibe)}}"</div>`;
                }}
                
                if (hasViolation) {{
                    const issues = [];
                    if (book.subject_count_violation) issues.push('Too many subjects (' + book.llm_subjects.length + ' > 8)');
                    if (book.has_duplicates) issues.push('Has duplicate subjects');
                    html += `<div class="warning">⚠️ VALIDATION ISSUE: ${{issues.join(' | ')}}</div>`;
                }}
                
                if (book.vibe_violation) {{
                    html += `<div class="warning">⚠️ Vibe length outside 8-12 words</div>`;
                }}
                
                html += `</div>`;
                container.innerHTML += html;
            }});
            
            document.getElementById('visibleCount').textContent = books.length;
        }}
        
        function filterBooks() {{
            const quality = document.getElementById('qualityFilter').value;
            const validation = document.getElementById('validationFilter').value;
            const errorFilter = document.getElementById('errorFilter').value;
            const genre = document.getElementById('genreFilter').value;
            const search = document.getElementById('searchBox').value.toLowerCase();
            
            let filtered = allBooks.filter(book => {{
                // Quality filter
                if (quality !== 'all' && book.metadata_quality !== quality) return false;
                
                // Validation filter
                if (validation === 'clean' && (book.has_duplicates || book.subject_count_violation || book.vibe_violation)) return false;
                if (validation === 'violations' && !(book.has_duplicates || book.subject_count_violation)) return false;
                if (validation === 'duplicates' && !book.has_duplicates) return false;
                if (validation === 'too-many' && !book.subject_count_violation) return false;
                if (validation === 'vibe-wrong' && !book.vibe_violation) return false;
                
                // Error filter
                if (errorFilter === 'no-errors' && book.has_error) return false;
                if (errorFilter === 'has-errors' && !book.has_error) return false;
                
                // Genre filter
                if (genre !== 'all' && book.genre !== genre) return false;
                
                // Search filter
                if (search && !book.title.toLowerCase().includes(search) && !book.author.toLowerCase().includes(search)) return false;
                
                return true;
            }});
            
            renderBooks(filtered);
        }}
        
        function resetFilters() {{
            document.getElementById('qualityFilter').value = 'all';
            document.getElementById('validationFilter').value = 'all';
            document.getElementById('errorFilter').value = 'all';
            document.getElementById('genreFilter').value = 'all';
            document.getElementById('searchBox').value = '';
            filterBooks();
        }}
        
        // Attach event listeners
        document.getElementById('qualityFilter').addEventListener('change', filterBooks);
        document.getElementById('validationFilter').addEventListener('change', filterBooks);
        document.getElementById('errorFilter').addEventListener('change', filterBooks);
        document.getElementById('genreFilter').addEventListener('change', filterBooks);
        document.getElementById('searchBox').addEventListener('input', filterBooks);
        
        // Initial render
        renderBooks(allBooks);
    </script>
</body>
</html>
"""
    
    return html_content


def main():
    parser = argparse.ArgumentParser(description="Analyze enrichment results from SQL and generate HTML report")
    parser.add_argument("--version", default="v2", help="Tags version (default: v2)")
    parser.add_argument("--limit", type=int, default=100, help="Number of books to analyze (0 = all)")
    args = parser.parse_args()
    
    print("="*80)
    print("ENRICHMENT RESULTS ANALYSIS")
    print("="*80)
    print(f"Tags version: {args.version}")
    if args.limit > 0:
        print(f"Analyzing up to {args.limit} books\n")
    else:
        print("Analyzing all enriched books\n")
    
    # Fetch enriched books
    print("Fetching enriched books from SQL...")
    books = fetch_enriched_books(args.version, args.limit if args.limit > 0 else 999999)
    print(f"Found {len(books)} enriched books\n")
    
    if not books:
        print("No enriched books found!")
        return
    
    # Analyze results
    print("Analyzing results...")
    stats = analyze_results(books)
    
    # Analyze errors
    error_breakdown, total_errors = analyze_errors(args.version)
    
    # Display quick stats (console)
    print("\n" + "="*80)
    print("QUICK STATISTICS")
    print("="*80)
    
    avg_subjects = sum(stats['subject_counts']) / len(stats['subject_counts'])
    avg_tones = sum(stats['tone_counts']) / len(stats['tone_counts'])
    avg_vibe = sum(stats['vibe_lengths']) / len(stats['vibe_lengths']) if stats['vibe_lengths'] else 0
    
    print(f"\nTotal books: {stats['total']}")
    print(f"Average subjects: {avg_subjects:.1f}")
    print(f"Average tones: {avg_tones:.1f}")
    print(f"Average vibe length: {avg_vibe:.1f} words")
    print(f"\nBooks with violations: {len(stats['subject_count_violations'])}")
    print(f"Books with duplicates: {stats['has_duplicates']}")
    print(f"Vibe length violations: {len(stats['vibe_length_violations'])}")
    print(f"Total errors: {total_errors}")
    
    # Generate HTML report
    print("\n" + "="*80)
    print("GENERATING INTERACTIVE HTML REPORT")
    print("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"enrichment_report_{args.version}_{timestamp}.html"
    
    print("\nGenerating report...")
    html_content = generate_html_report(books, stats, error_breakdown, total_errors, args.version)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ HTML report generated: {filename}")
    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print("="*80)
    print(f"\n1. Open the HTML report in your browser:")
    print(f"   {filename}")
    print(f"\n2. Use the filters to explore:")
    print(f"   • Filter by metadata quality (rich/medium/poor/none)")
    print(f"   • Filter by validation issues")
    print(f"   • Filter by genre")
    print(f"   • Search by title or author")
    print(f"\n3. Review all {len(books)} books with interactive browsing")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
