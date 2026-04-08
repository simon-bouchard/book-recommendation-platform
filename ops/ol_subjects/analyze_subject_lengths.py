# ops/ol_subjects/analyze_subject_lengths.py
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "ol_subjects" / "ol_subjects_cleaned_v5_combined.jsonl"

# Load all subjects
all_subjects = []
with open(INPUT, encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        subjects = rec.get("subjects", [])
        all_subjects.extend(subjects)

counter = Counter(all_subjects)

# Analyze by length
subjects_with_len = [(s, len(s), counter[s]) for s in counter.keys()]
subjects_with_len.sort(key=lambda x: x[1], reverse=True)

print("=" * 80)
print("LONGEST SUBJECTS")
print("=" * 80)
print("\nTop 50 longest subjects (char count | frequency | subject):")
for subj, length, freq in subjects_with_len[:50]:
    print(f"{length:>4} chars | {freq:>6}x | {subj[:100]}")

print("\n" + "=" * 80)
print("LENGTH DISTRIBUTION")
print("=" * 80)

# Buckets
buckets = {
    "1-20": 0,
    "21-50": 0,
    "51-100": 0,
    "101-150": 0,
    "151-200": 0,
    "201+": 0,
}

for subj, length, freq in subjects_with_len:
    if length <= 20:
        buckets["1-20"] += freq
    elif length <= 50:
        buckets["21-50"] += freq
    elif length <= 100:
        buckets["51-100"] += freq
    elif length <= 150:
        buckets["101-150"] += freq
    elif length <= 200:
        buckets["151-200"] += freq
    else:
        buckets["201+"] += freq

print("\nSubject instances by length:")
for bucket, count in buckets.items():
    print(f"  {bucket:>10} chars: {count:>8,} instances")

print("\n" + "=" * 80)
print("LONG + RARE SUBJECTS")
print("=" * 80)

# Find long subjects that are rare
long_rare = [(s, l, f) for s, l, f in subjects_with_len if l > 100 and f <= 5]
long_rare.sort(key=lambda x: x[1], reverse=True)

print(f"\nSubjects >100 chars appearing ≤5 times: {len(long_rare)}")
if long_rare:
    print("\nExamples:")
    for subj, length, freq in long_rare[:20]:
        print(f"  {length} chars | {freq}x | {subj[:100]}...")

# Stats on >100 char subjects
very_long = [x for x in subjects_with_len if x[1] > 100]
total_instances = sum(x[2] for x in very_long)
unique_count = len(very_long)

print("\nSubjects >100 chars:")
print(f"  Unique: {unique_count}")
print(f"  Total instances: {total_instances:,}")
print(f"  Avg frequency: {total_instances / unique_count:.1f}")

print("\n" + "=" * 80)
