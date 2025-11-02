#!/usr/bin/env python3
"""
Simple error analysis - statistics on validation failures.

Usage:
    python ops/enrichment/analyze_test_errors.py test_errors_20250130_023456.txt
"""
import sys
import json
import re
import argparse
from pathlib import Path
from collections import Counter

OUTPUT_DIR = Path(__file__).parent


def parse_error_file(error_file_path):
    """Parse the error text file into structured data."""
    with open(error_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    errors = []
    
    # Split by error blocks (separated by dashed lines)
    blocks = content.split('-' * 80)
    
    for block in blocks:
        if not block.strip() or 'item_idx:' not in block:
            continue
        
        error = {}
        lines = block.strip().split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            if line.startswith('item_idx:'):
                error['item_idx'] = line.split(':', 1)[1].strip()
            elif line.startswith('title:'):
                error['title'] = line.split(':', 1)[1].strip()
            elif line.startswith('author:'):
                error['author'] = line.split(':', 1)[1].strip()
            elif line.startswith('error_msg:'):
                error['error_msg'] = line.split(':', 1)[1].strip()
            elif line.startswith('stage:'):
                error['stage'] = line.split(':', 1)[1].strip()
            elif line.startswith('attempted_response:'):
                # Parse JSON that follows
                json_start = i + 1
                json_lines = []
                i += 1
                while i < len(lines) and lines[i].strip():
                    json_lines.append(lines[i])
                    i += 1
                
                if json_lines:
                    try:
                        json_str = '\n'.join(json_lines)
                        error['attempted'] = json.loads(json_str)
                    except:
                        error['attempted'] = None
                continue
            
            i += 1
        
        if error.get('item_idx'):
            errors.append(error)
    
    return errors


def classify_validation_error(error_msg):
    """Classify validation error into a category."""
    msg = error_msg.lower()
    
    if 'vibe too short' in msg or 'vibe_too_short' in msg:
        return 'VIBE_TOO_SHORT'
    elif 'vibe too long' in msg or 'vibe_too_long' in msg:
        return 'VIBE_TOO_LONG'
    elif 'near-duplicate' in msg:
        return 'NEAR_DUPLICATE_SUBJECTS'
    elif 'subject' in msg and 'count' in msg:
        if 'below' in msg or 'minimum' in msg:
            return 'SUBJECT_COUNT_TOO_LOW'
        elif 'exceed' in msg or 'maximum' in msg:
            return 'SUBJECT_COUNT_TOO_HIGH'
        else:
            return 'SUBJECT_COUNT_WRONG'
    elif 'tone' in msg and 'count' in msg:
        if 'below' in msg or 'minimum' in msg:
            return 'TONE_COUNT_TOO_LOW'
        elif 'exceed' in msg or 'maximum' in msg:
            return 'TONE_COUNT_TOO_HIGH'
        else:
            return 'TONE_COUNT_WRONG'
    elif 'invalid genre' in msg:
        return 'INVALID_GENRE'
    elif 'invalid tone' in msg:
        return 'INVALID_TONE_ID'
    else:
        return 'OTHER_VALIDATION'


def count_words(text):
    """Count words the same way validator does (split on spaces)."""
    if not text:
        return 0
    return len(text.split())


def main():
    parser = argparse.ArgumentParser(description="Analyze enrichment test errors")
    parser.add_argument("error_file", help="Path to test errors txt file")
    args = parser.parse_args()
    
    error_path = Path(args.error_file)
    if not error_path.exists():
        print(f"Error: File not found: {error_path}")
        sys.exit(1)
    
    print("Parsing errors...\n")
    all_errors = parse_error_file(error_path)
    
    # Filter to only validation errors
    validation_errors = [e for e in all_errors if e.get('stage') == 'validate']
    
    print("="*80)
    print("VALIDATION ERROR BREAKDOWN")
    print("="*80)
    print(f"\nTotal validation errors: {len(validation_errors)}")
    print(f"Total other errors: {len(all_errors) - len(validation_errors)}\n")
    
    # Classify each validation error
    error_types = Counter()
    errors_by_type = {}
    
    for err in validation_errors:
        error_type = classify_validation_error(err.get('error_msg', ''))
        error_types[error_type] += 1
        
        if error_type not in errors_by_type:
            errors_by_type[error_type] = []
        errors_by_type[error_type].append(err)
    
    # Show breakdown
    print("Validation Error Types:")
    print("-"*80)
    for error_type, count in error_types.most_common():
        pct = (count / len(validation_errors) * 100) if validation_errors else 0
        print(f"  {error_type:35s} {count:5d} ({pct:5.1f}%)")
    
    # Detailed analysis for each type
    for error_type, error_list in sorted(errors_by_type.items(), key=lambda x: len(x[1]), reverse=True):
        print("\n" + "="*80)
        print(f"{error_type} - {len(error_list)} cases")
        print("="*80)
        
        # Show 3-5 examples
        num_examples = min(5, len(error_list))
        print(f"\nShowing {num_examples} examples:\n")
        
        for i, err in enumerate(error_list[:num_examples], 1):
            print(f"Example {i}:")
            print(f"  Item: #{err['item_idx']}")
            print(f"  Title: {err['title'][:70]}")
            print(f"  Error: {err['error_msg'][:120]}")
            
            attempted = err.get('attempted')
            if attempted and isinstance(attempted, dict):
                tier = attempted.get('tier', '?')
                raw = attempted.get('raw_response', {})
                
                print(f"  Tier: {tier}")
                
                if isinstance(raw, dict):
                    # Show relevant data based on error type
                    if 'VIBE' in error_type:
                        vibe = raw.get('vibe', '')
                        word_count = count_words(vibe)
                        print(f"  Vibe ({word_count} words): \"{vibe}\"")
                        
                        # Check for hyphens
                        if vibe and '-' in vibe:
                            hyphenated = [w for w in vibe.split() if '-' in w]
                            print(f"  ⚠️  Has hyphens: {hyphenated}")
                    
                    elif 'SUBJECT' in error_type:
                        subjects = raw.get('subjects', [])
                        print(f"  Subjects ({len(subjects)}): {subjects}")
                    
                    elif 'TONE' in error_type:
                        tones = raw.get('tone_ids', [])
                        print(f"  Tone IDs ({len(tones)}): {tones}")
                    
                    elif 'GENRE' in error_type:
                        genre = raw.get('genre', '')
                        print(f"  Genre: \"{genre}\"")
            
            print()
        
        if len(error_list) > num_examples:
            print(f"  ... and {len(error_list) - num_examples} more similar cases\n")
    
    # Key insights
    print("="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print()
    
    # Vibe hyphen analysis
    if 'VIBE_TOO_SHORT' in errors_by_type:
        vibe_short = errors_by_type['VIBE_TOO_SHORT']
        hyphen_count = 0
        
        for e in vibe_short:
            attempted = e.get('attempted')
            if attempted and isinstance(attempted, dict):
                raw = attempted.get('raw_response', {})
                if isinstance(raw, dict):
                    vibe = raw.get('vibe', '')
                    if vibe and '-' in vibe:
                        hyphen_count += 1
        
        if hyphen_count > 0:
            print(f"• {hyphen_count}/{len(vibe_short)} short vibe errors contain hyphens")
            print("  → May need to clarify word counting in prompt")
    
    # Subject specificity
    if 'NEAR_DUPLICATE_SUBJECTS' in errors_by_type and len(errors_by_type['NEAR_DUPLICATE_SUBJECTS']) > 3:
        count = len(errors_by_type['NEAR_DUPLICATE_SUBJECTS'])
        print(f"• {count} near-duplicate subject errors")
        print("  → Prompt may need more emphasis on subject distinctiveness")
    
    # Count issues
    count_errors = sum(1 for k in error_types.keys() if 'COUNT' in k)
    if count_errors > 10:
        print(f"• {count_errors} count-related errors across subjects/tones")
        print("  → Check if tier requirements are clear in prompt")
    
    # Top issue
    if error_types:
        top_error, top_count = error_types.most_common(1)[0]
        pct = (top_count / len(validation_errors) * 100) if validation_errors else 0
        print(f"\n• Top issue: {top_error} ({top_count} cases, {pct:.1f}% of validation errors)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
