
# ops/ol_subjects/clean_subjects_v2.py
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT_JSONL = ROOT / "data" / "ol_subjects" / "ol_subjects_cleaned_v1.jsonl"
OUTPUT_JSONL = ROOT / "data" / "ol_subjects" / "ol_subjects_cleaned_v2.jsonl"

stats = {
    "removed_translation": 0,
    "stripped_works_by": 0,
    "split_in_art": 0,
    "split_in_literature": 0,
    "split_study_guide": 0,
    "split_juvenile_literature": 0,
}

def process_subject(s):
    if "translation" in s:
        stats["removed_translation"] += 1
        return []
    
    if s == "in literature" or s == "in art":
        stats["removed_translation"] += 1
        return []
    
    if "works by" in s:
        s = re.sub(r'works by.*$', '', s).strip()
        stats["stripped_works_by"] += 1
        if not s:
            return []
    
    match = re.match(r'^(.+?)\s+in art$', s)
    if match:
        x = match.group(1).strip()
        if x and x != "in art":
            stats["split_in_art"] += 1
            return [x, "art"]
    
    match = re.match(r'^(.+?)\s+in literature$', s)
    if match:
        x = match.group(1).strip()
        if x and x != "in literature" and not x.endswith(" literature"):
            stats["split_in_literature"] += 1
            return [x, "literature"]
    
    match = re.match(r'^(.+?)\s+study guides?$', s)
    if match:
        x = match.group(1).strip()
        if x:
            stats["split_study_guide"] += 1
            return [x, "study guide"]
    
    match = re.match(r'^(.+?)\s*juvenile literature$', s)
    if match:
        x = match.group(1).strip()
        if x:
            stats["split_juvenile_literature"] += 1
            return [x, "juvenile literature"]
    
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
print(f"\nTransformations:")
print(f"  Removed translations: {stats['removed_translation']:,}")
print(f"  Stripped 'works by': {stats['stripped_works_by']:,}")
print(f"  Split 'X in art': {stats['split_in_art']:,}")
print(f"  Split 'X in literature': {stats['split_in_literature']:,}")
print(f"  Split 'study guide X': {stats['split_study_guide']:,}")
print(f"  Split 'juvenile literature X': {stats['split_juvenile_literature']:,}")
print(f"\nSubject counts:")
print(f"  Before: {total_in:,}")
print(f"  After: {total_out:,}")
print(f"  Change: {total_out - total_in:,} ({(total_out/total_in - 1)*100:+.1f}%)")
print(f"\nOutput: {OUTPUT_JSONL}")
