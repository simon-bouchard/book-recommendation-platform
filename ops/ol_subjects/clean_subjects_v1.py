# ops/ol_subjects/clean_subjects_v1.py
import json
import re
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[2]
INPUT_JSONL = ROOT / "data" / "ol_subjects" / "ol_subjects.jsonl"
OUTPUT_JSONL = ROOT / "data" / "ol_subjects" / "ol_subjects_cleaned_v1.jsonl"

OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)


def clean_subject(s):
    """Clean a single subject string."""
    # Strip whitespace
    s = s.strip()
    if not s:
        return None

    # Remove unwanted characters (keep: letters, numbers, spaces, hyphens, apostrophes, ampersands)
    s = re.sub(r"[^a-zA-Z0-9\s\-'&]", "", s)

    # Lowercase
    s = s.lower()

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)

    s = s.strip()

    return s if s else None


def process_subjects(subjects):
    """Process a list of subjects: split, clean, dedupe."""
    if not isinstance(subjects, list):
        return []

    cleaned = []

    for subj in subjects:
        if not isinstance(subj, str):
            continue

        # Split on ' -- '
        parts = subj.split(" -- ")

        for part in parts:
            # Split on comma
            subparts = part.split(",")

            for subpart in subparts:
                cleaned_subj = clean_subject(subpart)
                if cleaned_subj:
                    cleaned.append(cleaned_subj)

    # Deduplicate (case-insensitive, but already lowercased)
    cleaned = list(dict.fromkeys(cleaned))

    return cleaned


# Process
total = 0
cleaned_total = 0

with (
    open(INPUT_JSONL, encoding="utf-8") as infile,
    open(OUTPUT_JSONL, "w", encoding="utf-8") as outfile,
):
    for line in infile:
        record = json.loads(line)
        work_id = record.get("work_id")
        subjects = record.get("subjects", [])

        original_count = len(subjects) if isinstance(subjects, list) else 0
        cleaned_subjects = process_subjects(subjects)

        total += original_count
        cleaned_total += len(cleaned_subjects)

        outfile.write(
            json.dumps({"work_id": work_id, "subjects": cleaned_subjects}, ensure_ascii=False)
            + "\n"
        )

print(f"Cleaned {INPUT_JSONL.name}")
print(f"  Original subjects: {total:,}")
print(f"  Cleaned subjects: {cleaned_total:,}")
print(f"  Reduction: {total - cleaned_total:,} ({(1 - cleaned_total / total) * 100:.1f}%)")
print(f"  Output: {OUTPUT_JSONL}")
