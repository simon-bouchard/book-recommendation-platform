# ops/ol_subjects/clean_subjects_v3.py
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT_JSONL = ROOT / "data" / "ol_subjects" / "ol_subjects_cleaned_v2.jsonl"
OUTPUT_JSONL = ROOT / "data" / "ol_subjects" / "ol_subjects_cleaned_v3.jsonl"

stats = {
    "removed_fictitious": 0,
    "removed_reviewed": 0,
    "removed_award": 0,
}

def process_subject(s):
    # Remove: fictitious character
    if "fictitious character" in s:
        stats["removed_fictitious"] += 1
        return []
    
    # Remove: reviewed (any review mentions)
    if "reviewed" in s:
        stats["removed_reviewed"] += 1
        return []
    
    # Remove: award
    if "award" in s:
        stats["removed_award"] += 1
        return []
    
    return [s]


def process_subjects(subjects):
    if not isinstance(subjects, list):
        return []
    
    result = []
    for subj in subjects:
        if not isinstance(subj, str):
            continue
        processed = process_subject(subj)
        result.extend(processed)
    
    result = list(dict.fromkeys(result))
    return result


total_in = 0
total_out = 0

with open(INPUT_JSONL, encoding="utf-8") as infile, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as outfile:
    
    for line in infile:
        record = json.loads(line)
        work_id = record.get("work_id")
        subjects = record.get("subjects", [])
        
        original_count = len(subjects) if isinstance(subjects, list) else 0
        cleaned_subjects = process_subjects(subjects)
        
        total_in += original_count
        total_out += len(cleaned_subjects)
        
        outfile.write(json.dumps({
            "work_id": work_id,
            "subjects": cleaned_subjects
        }, ensure_ascii=False) + "\n")

print(f"Cleaned {INPUT_JSONL.name}")
print(f"\nRemovals:")
print(f"  Fictitious character: {stats['removed_fictitious']:,}")
print(f"  Reviewed: {stats['removed_reviewed']:,}")
print(f"  Award: {stats['removed_award']:,}")
print(f"\nSubject counts:")
print(f"  Before: {total_in:,}")
print(f"  After: {total_out:,}")
print(f"  Removed: {total_in - total_out:,} ({(1 - total_out/total_in)*100:.1f}%)")
print(f"\nOutput: {OUTPUT_JSONL}")
