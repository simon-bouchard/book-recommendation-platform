
#!/usr/bin/env python3
"""
Analyze enrichment test errors and generate an interactive HTML report.

Usage:
    python ops/enrichment/analyze_test_errors.py test_errors_20250130_023456.txt
    
Generates:
    - errors_analysis_<timestamp>.html - Interactive HTML report
"""
import sys
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

OUTPUT_DIR = Path(__file__).parent


def parse_error_file(error_file_path):
    """Parse the error text file into structured data."""
    with open(error_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract frequency table
    freq_section = re.search(r'Error Frequency:\n-+\n(.*?)\n\n', content, re.DOTALL)
    frequencies = {}
    if freq_section:
        for line in freq_section.group(1).split('\n'):
            match = re.match(r'(\S+.*?)\s+\|\s+(\d+)', line)
            if match:
                error_code = match.group(1).strip()
                count = int(match.group(2))
                frequencies[error_code] = count
    
    # Extract individual error examples
    errors = []
    error_blocks = re.findall(
        r'-{80}\n(.*?)\n(?=-{80}|\n\n\n|$)',
        content,
        re.DOTALL
    )
    
    for block in error_blocks:
        error = {}
        
        # Parse fields
        for line in block.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                if key == 'attempted_response':
                    # Try to parse JSON
                    try:
                        json_match = re.search(r'\{.*\}', block, re.DOTALL)
                        if json_match:
                            error['attempted_response'] = json.loads(json_match.group(0))
                    except:
                        error['attempted_response'] = value
                else:
                    error[key] = value
        
        if error.get('item_idx'):
            errors.append(error)
    
    return frequencies, errors


def generate_html_report(frequencies, errors, output_path):
    """Generate interactive HTML report for errors."""
    
    # Group errors by code
    by_error_code = defaultdict(list)
    for err in errors:
        code = err.get('error_code', 'UNKNOWN')
        by_error_code[code].append(err)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Enrichment Errors Analysis</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #f44336;
            padding-bottom: 10px;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        .freq-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        .freq-table th {{
            background: #f44336;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        .freq-table td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        .freq-table tr:hover {{
            background: #f5f5f5;
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
        .error-group {{
            margin: 30px 0;
        }}
        .error-group-header {{
            background: #f44336;
            color: white;
            padding: 15px;
            border-radius: 8px 8px 0 0;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .error-group-header:hover {{
            background: #d32f2f;
        }}
        .error-group-content {{
            background: white;
            border-radius: 0 0 8px 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .error-group-content.collapsed {{
            display: none;
        }}
        .error-card {{
            padding: 20px;
            border-bottom: 1px solid #eee;
        }}
        .error-card:last-child {{
            border-bottom: none;
        }}
        .error-card.hidden {{
            display: none;
        }}
        .error-header {{
            display: flex;
            justify-content: space-between;
            align-items: start;
            margin-bottom: 15px;
        }}
        .book-title {{
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin: 0 0 5px 0;
        }}
        .book-author {{
            font-size: 14px;
            color: #666;
            margin: 0;
        }}
        .item-idx {{
            background: #ffebee;
            color: #c62828;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }}
        .error-info {{
            background: #fff3e0;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            border-left: 4px solid #ff9800;
        }}
        .error-label {{
            font-weight: bold;
            color: #e65100;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .error-value {{
            color: #333;
            margin-top: 3px;
        }}
        .section {{
            margin: 15px 0;
        }}
        .section-title {{
            font-weight: bold;
            color: #666;
            margin-bottom: 5px;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .section-content {{
            color: #333;
            line-height: 1.6;
            background: #fafafa;
            padding: 10px;
            border-radius: 4px;
        }}
        .json-view {{
            background: #263238;
            color: #aed581;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.5;
        }}
        .json-key {{
            color: #82aaff;
        }}
        .json-string {{
            color: #c3e88d;
        }}
        .json-number {{
            color: #f78c6c;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            margin-right: 5px;
        }}
        .badge.stage {{
            background: #e1f5fe;
            color: #01579b;
        }}
        .badge.tier {{
            background: #f3e5f5;
            color: #4a148c;
        }}
        .toggle-icon {{
            font-size: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Enrichment Errors Analysis</h1>
        
        <div class="summary">
            <h2>Error Frequency Summary</h2>
            <table class="freq-table">
                <thead>
                    <tr>
                        <th>Error Code</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    total_errors = sum(frequencies.values())
    for error_code, count in sorted(frequencies.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_errors * 100) if total_errors > 0 else 0
        html += f"""                    <tr>
                        <td><strong>{error_code}</strong></td>
                        <td>{count}</td>
                        <td>{pct:.1f}%</td>
                    </tr>
"""
    
    html += f"""                </tbody>
            </table>
            <p><strong>Total Errors:</strong> {total_errors}</p>
        </div>
        
        <div class="filters">
            <label>Search: <input type="text" id="searchBox" placeholder="Search by title, error message..."></label>
            <label>Error Code: 
                <select id="errorCodeFilter">
                    <option value="">All Error Codes</option>
"""
    
    for error_code in sorted(frequencies.keys()):
        html += f'                    <option value="{error_code}">{error_code}</option>\n'
    
    html += """                </select>
            </label>
        </div>
        
        <div id="errorList">
"""
    
    # Group errors by code
    for error_code in sorted(by_error_code.keys(), key=lambda x: frequencies.get(x, 0), reverse=True):
        errors_list = by_error_code[error_code]
        count = len(errors_list)
        
        html += f"""
            <div class="error-group">
                <div class="error-group-header" onclick="toggleGroup(this)">
                    <div>
                        <strong>{error_code}</strong> 
                        <span style="opacity: 0.8;">({count} occurrences)</span>
                    </div>
                    <span class="toggle-icon">▼</span>
                </div>
                <div class="error-group-content" data-error-code="{error_code}">
"""
        
        # Show individual errors
        for err in errors_list:
            attempted = err.get('attempted_response', {})
            tier = 'UNKNOWN'
            if isinstance(attempted, dict):
                tier = attempted.get('tier', 'UNKNOWN')
            
            html += f"""
                    <div class="error-card">
                        <div class="error-header">
                            <div>
                                <h3 class="book-title">{err.get('title', 'Unknown')}</h3>
                                <p class="book-author">{err.get('author', 'Unknown')}</p>
                            </div>
                            <span class="item-idx">#{err.get('item_idx', '?')}</span>
                        </div>
                        
                        <div style="margin: 10px 0;">
                            <span class="badge stage">{err.get('stage', 'unknown')}</span>
                            <span class="badge tier">{tier}</span>
                        </div>
                        
                        <div class="error-info">
                            <div class="error-label">Error Message</div>
                            <div class="error-value">{err.get('error_msg', 'No message')}</div>
                        </div>
                        
                        <div class="section">
                            <div class="section-title">Description</div>
                            <div class="section-content">{err.get('description', 'No description')}</div>
                        </div>
"""
            
            # Show attempted response if available
            if attempted and isinstance(attempted, dict):
                # Format the JSON nicely
                json_str = json.dumps(attempted, indent=2, ensure_ascii=False)
                # Simple syntax highlighting
                json_str = json_str.replace('"subjects"', '<span class="json-key">"subjects"</span>')
                json_str = json_str.replace('"tone_ids"', '<span class="json-key">"tone_ids"</span>')
                json_str = json_str.replace('"genre"', '<span class="json-key">"genre"</span>')
                json_str = json_str.replace('"vibe"', '<span class="json-key">"vibe"</span>')
                json_str = json_str.replace('"tier"', '<span class="json-key">"tier"</span>')
                json_str = json_str.replace('"score"', '<span class="json-key">"score"</span>')
                json_str = json_str.replace('"raw_response"', '<span class="json-key">"raw_response"</span>')
                
                html += f"""
                        <div class="section">
                            <div class="section-title">Attempted Response (What LLM Tried)</div>
                            <div class="json-view">{json_str}</div>
                        </div>
"""
            
            html += """                    </div>
"""
        
        html += """                </div>
            </div>
"""
    
    html += """        </div>
    </div>
    
    <script>
        function toggleGroup(header) {
            const content = header.nextElementSibling;
            const icon = header.querySelector('.toggle-icon');
            content.classList.toggle('collapsed');
            icon.textContent = content.classList.contains('collapsed') ? '▶' : '▼';
        }
        
        // Filter functionality
        const searchBox = document.getElementById('searchBox');
        const errorCodeFilter = document.getElementById('errorCodeFilter');
        
        function filterErrors() {
            const searchTerm = searchBox.value.toLowerCase();
            const selectedCode = errorCodeFilter.value;
            
            // Filter groups
            document.querySelectorAll('.error-group').forEach(group => {
                const content = group.querySelector('.error-group-content');
                const groupCode = content.dataset.errorCode;
                
                if (selectedCode && groupCode !== selectedCode) {
                    group.style.display = 'none';
                } else {
                    group.style.display = 'block';
                }
            });
            
            // Filter individual cards
            document.querySelectorAll('.error-card').forEach(card => {
                const text = card.textContent.toLowerCase();
                
                if (searchTerm && !text.includes(searchTerm)) {
                    card.classList.add('hidden');
                } else {
                    card.classList.remove('hidden');
                }
            });
        }
        
        searchBox.addEventListener('input', filterErrors);
        errorCodeFilter.addEventListener('change', filterErrors);
    </script>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✓ Generated HTML report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze enrichment test errors")
    parser.add_argument("error_file", help="Path to test errors txt file")
    args = parser.parse_args()
    
    error_path = Path(args.error_file)
    if not error_path.exists():
        print(f"Error: File not found: {error_path}")
        sys.exit(1)
    
    print(f"Parsing errors from {error_path}...")
    frequencies, errors = parse_error_file(error_path)
    
    print(f"Found {len(errors)} error examples across {len(frequencies)} error codes")
    
    # Generate HTML report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"errors_analysis_{timestamp}.html"
    
    print("Generating HTML report...")
    generate_html_report(frequencies, errors, output_path)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Total errors: {sum(frequencies.values())}")
    print("\nTop errors:")
    for error_code, count in sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {error_code}: {count}")
    print(f"\nOpen the HTML file in your browser:")
    print(f"  {output_path}")


if __name__ == "__main__":
    main()
