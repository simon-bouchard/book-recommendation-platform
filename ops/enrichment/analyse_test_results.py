#!/usr/bin/env python3
"""
Analyze enrichment test results and generate an HTML report.

Usage:
    python ops/enrichment/analyze_test_results.py test_results_20250101_120000.csv
    
Generates:
    - results_analysis_<timestamp>.html - Interactive HTML report
"""
import sys
import csv
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

OUTPUT_DIR = Path(__file__).parent


def load_results(csv_path):
    """Load results from CSV."""
    results = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    
    return results


def analyze_stats(results):
    """Generate basic statistics."""
    stats = {
        'total': len(results),
        'by_tier': Counter(),
        'vibe_lengths': [],
        'subject_counts': [],
        'tone_counts': [],
        'genres': Counter(),
    }
    
    for r in results:
        # Count by tier if available (might not be in CSV)
        tier = r.get('tier', 'UNKNOWN')
        stats['by_tier'][tier] += 1
        
        # Vibe length
        vibe = r.get('vibe', '')
        if vibe:
            word_count = len(vibe.split())
            stats['vibe_lengths'].append(word_count)
        
        # Subject count
        subjects = r.get('llm_subjects', '')
        if subjects:
            subject_count = len([s.strip() for s in subjects.split(';') if s.strip()])
            stats['subject_counts'].append(subject_count)
        
        # Tone count
        tones = r.get('tone_ids', '')
        if tones:
            tone_count = len([t.strip() for t in tones.split(',') if t.strip()])
            stats['tone_counts'].append(tone_count)
        
        # Genre
        genre = r.get('genre', '')
        if genre:
            stats['genres'][genre] += 1
    
    return stats


def generate_html_report(results, stats, output_path):
    """Generate interactive HTML report."""
    
    # Calculate averages
    avg_vibe_len = sum(stats['vibe_lengths']) / len(stats['vibe_lengths']) if stats['vibe_lengths'] else 0
    avg_subjects = sum(stats['subject_counts']) / len(stats['subject_counts']) if stats['subject_counts'] else 0
    avg_tones = sum(stats['tone_counts']) / len(stats['tone_counts']) if stats['tone_counts'] else 0
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Enrichment Results Analysis</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            color: #666;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .stat-card .value {{
            font-size: 32px;
            font-weight: bold;
            color: #4CAF50;
        }}
        .filters {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .filters input, .filters select {{
            padding: 8px;
            margin: 5px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .book-card {{
            background: white;
            padding: 20px;
            margin: 15px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #4CAF50;
        }}
        .book-card.hidden {{
            display: none;
        }}
        .book-header {{
            display: flex;
            justify-content: space-between;
            align-items: start;
            margin-bottom: 15px;
        }}
        .book-title {{
            font-size: 20px;
            font-weight: bold;
            color: #333;
            margin: 0 0 5px 0;
        }}
        .book-author {{
            font-size: 16px;
            color: #666;
            margin: 0;
        }}
        .item-idx {{
            background: #f0f0f0;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            color: #666;
        }}
        .section {{
            margin: 15px 0;
        }}
        .section-title {{
            font-weight: bold;
            color: #4CAF50;
            margin-bottom: 5px;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .section-content {{
            color: #333;
            line-height: 1.6;
        }}
        .tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 5px;
        }}
        .tag {{
            background: #e3f2fd;
            color: #1976d2;
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 13px;
        }}
        .tag.ol-subject {{
            background: #f3e5f5;
            color: #7b1fa2;
        }}
        .tag.tone {{
            background: #fff3e0;
            color: #e65100;
        }}
        .description {{
            background: #fafafa;
            padding: 10px;
            border-radius: 4px;
            border-left: 3px solid #ddd;
            font-size: 14px;
            color: #555;
            font-style: italic;
        }}
        .genre-badge {{
            display: inline-block;
            background: #4CAF50;
            color: white;
            padding: 6px 12px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Enrichment Results Analysis</h1>
        
        <div class="stats">
            <div class="stat-card">
                <h3>Total Books</h3>
                <div class="value">{stats['total']}</div>
            </div>
            <div class="stat-card">
                <h3>Avg Vibe Length</h3>
                <div class="value">{avg_vibe_len:.1f}</div>
            </div>
            <div class="stat-card">
                <h3>Avg Subjects</h3>
                <div class="value">{avg_subjects:.1f}</div>
            </div>
            <div class="stat-card">
                <h3>Avg Tones</h3>
                <div class="value">{avg_tones:.1f}</div>
            </div>
        </div>
        
        <div class="filters">
            <label>Search: <input type="text" id="searchBox" placeholder="Search by title, author, subject..."></label>
            <label>Genre: 
                <select id="genreFilter">
                    <option value="">All Genres</option>
"""
    
    # Add genre options
    for genre, count in sorted(stats['genres'].items()):
        html += f'                    <option value="{genre}">{genre} ({count})</option>\n'
    
    html += """                </select>
            </label>
            <label>
                <input type="checkbox" id="shortVibesOnly"> Short vibes only (< 8 words)
            </label>
            <label>
                <input type="checkbox" id="longVibesOnly"> Long vibes only (> 12 words)
            </label>
        </div>
        
        <div id="bookList">
"""
    
    # Add each book
    for r in results:
        subjects = [s.strip() for s in r.get('llm_subjects', '').split(';') if s.strip()]
        ol_subjects = [s.strip() for s in r.get('ol_subjects', '').split(';') if s.strip()]
        tones = [t.strip() for t in r.get('tone_ids', '').split(',') if t.strip()]
        vibe = r.get('vibe', '')
        vibe_len = len(vibe.split()) if vibe else 0
        
        html += f"""
            <div class="book-card" data-genre="{r.get('genre', '')}" data-vibe-len="{vibe_len}">
                <div class="book-header">
                    <div>
                        <h2 class="book-title">{r.get('title', 'Unknown')}</h2>
                        <p class="book-author">{r.get('author', 'Unknown')}</p>
                    </div>
                    <span class="item-idx">#{r.get('item_idx', '?')}</span>
                </div>
                
                <div class="section">
                    <div class="section-title">Description</div>
                    <div class="description">{r.get('description', 'No description')}</div>
                </div>
                
                <div class="section">
                    <div class="section-title">Genre</div>
                    <span class="genre-badge">{r.get('genre', 'none')}</span>
                </div>
                
                <div class="section">
                    <div class="section-title">OpenLibrary Subjects ({len(ol_subjects)})</div>
                    <div class="tags">
"""
        
        for subj in ol_subjects:
            html += f'                        <span class="tag ol-subject">{subj}</span>\n'
        
        html += f"""                    </div>
                </div>
                
                <div class="section">
                    <div class="section-title">LLM Subjects ({len(subjects)})</div>
                    <div class="tags">
"""
        
        for subj in subjects:
            html += f'                        <span class="tag">{subj}</span>\n'
        
        html += f"""                    </div>
                </div>
                
                <div class="section">
                    <div class="section-title">Tones ({len(tones)})</div>
                    <div class="tags">
"""
        
        for tone in tones:
            html += f'                        <span class="tag tone">{tone}</span>\n'
        
        html += f"""                    </div>
                </div>
                
                <div class="section">
                    <div class="section-title">Vibe ({vibe_len} words)</div>
                    <div class="section-content">{vibe or 'No vibe'}</div>
                </div>
            </div>
"""
    
    html += """        </div>
    </div>
    
    <script>
        // Filter functionality
        const searchBox = document.getElementById('searchBox');
        const genreFilter = document.getElementById('genreFilter');
        const shortVibesOnly = document.getElementById('shortVibesOnly');
        const longVibesOnly = document.getElementById('longVibesOnly');
        const bookCards = document.querySelectorAll('.book-card');
        
        function filterBooks() {
            const searchTerm = searchBox.value.toLowerCase();
            const selectedGenre = genreFilter.value;
            const showShortOnly = shortVibesOnly.checked;
            const showLongOnly = longVibesOnly.checked;
            
            bookCards.forEach(card => {
                const text = card.textContent.toLowerCase();
                const genre = card.dataset.genre;
                const vibeLen = parseInt(card.dataset.vibeLen);
                
                let show = true;
                
                // Search filter
                if (searchTerm && !text.includes(searchTerm)) {
                    show = false;
                }
                
                // Genre filter
                if (selectedGenre && genre !== selectedGenre) {
                    show = false;
                }
                
                // Vibe length filters
                if (showShortOnly && vibeLen >= 8) {
                    show = false;
                }
                if (showLongOnly && vibeLen <= 12) {
                    show = false;
                }
                
                card.classList.toggle('hidden', !show);
            });
        }
        
        searchBox.addEventListener('input', filterBooks);
        genreFilter.addEventListener('change', filterBooks);
        shortVibesOnly.addEventListener('change', filterBooks);
        longVibesOnly.addEventListener('change', filterBooks);
    </script>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✓ Generated HTML report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze enrichment test results")
    parser.add_argument("csv_file", help="Path to test results CSV")
    args = parser.parse_args()
    
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    print(f"Loading results from {csv_path}...")
    results = load_results(csv_path)
    print(f"Loaded {len(results)} results")
    
    print("Analyzing statistics...")
    stats = analyze_stats(results)
    
    # Generate HTML report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"results_analysis_{timestamp}.html"
    
    print("Generating HTML report...")
    generate_html_report(results, stats, output_path)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Total books: {stats['total']}")
    print(f"Average vibe length: {sum(stats['vibe_lengths']) / len(stats['vibe_lengths']) if stats['vibe_lengths'] else 0:.1f} words")
    print(f"Average subjects: {sum(stats['subject_counts']) / len(stats['subject_counts']) if stats['subject_counts'] else 0:.1f}")
    print(f"Average tones: {sum(stats['tone_counts']) / len(stats['tone_counts']) if stats['tone_counts'] else 0:.1f}")
    print(f"\nOpen the HTML file in your browser:")
    print(f"  {output_path}")


if __name__ == "__main__":
    main()
