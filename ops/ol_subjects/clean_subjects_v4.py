# ops/ol_subjects/clean_subjects_v4.py
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT_JSONL = ROOT / "data" / "ol_subjects" / "ol_subjects_cleaned_v3.jsonl"
OUTPUT_JSONL = ROOT / "data" / "ol_subjects" / "ol_subjects_cleaned_v4.jsonl"

# Mapping: subject → normalized subject (None means remove)
SUBJECT_MAPPING = {
    # --------------------------
    # Truly meaningless / remove
    # --------------------------
    "general": None,
    "etc": None,
    "etc pour la jeunesse": None,
    "new york times bestseller": None,
    "open library staff picks": None,
    "bestsellers": None,
    "best books of the year": None,
    # --------------------------
    # Existing mappings (kept from previous file)
    # --------------------------
    "ficcin": "fiction",
    "novela": "fiction",
    "roman": "fiction",
    "romans": "fiction",
    "nouvelles": "short stories",
    "histoire": "history",
    "juvenile literature": "juvenile fiction",
    "juvenile": "juvenile fiction",
    "ficcin juvenil": "juvenile fiction",
    "children's stories": "children's fiction",
    "detective and mystery stories": "mystery & detective",
    "mystery and detective stories": "mystery & detective",
    "mystery": "mystery & detective",
    "mystery fiction": "mystery & detective",
    "romance fiction": "romance",
    "historical": "historical fiction",
    "horror tales": "horror",
    "horror stories": "horror",
    "humorous": "humor",
    "humorous stories": "humor",
    "american wit and humor": "humor",
    "short stories single author": "short stories",
    "american science fiction": "science fiction",
    "fantasy fiction": "fantasy",
    "thrillers": "thriller",
    "biographies": "biography",
    "poetry poetic": "poetry",
    "american fiction fictional": "american fiction",
    "british and irish fiction fictional": "british and irish fiction",
    "continental european fiction fictional": "continental european fiction",
    "comic books": "comics & graphic novels",
    "strips": "comics & graphic novels",
    "cartoons and comics": "comics & graphic novels",
    "psychological": "psychological fiction",
    "adventure and adventurers": "adventure",
    "adventure stories": "adventure",
    "action & adventure": "adventure",
    # --------------------------
    # New language variants / translations (added)
    # --------------------------
    "novela juvenil": "juvenile fiction",
    "chang pian xiao shuo": "fiction",
    "historia": "history",
    "murs et coutumes": "manners and customs",
    "wit and humor": "humor",
    # --------------------------
    # British/American variants combined to generic (added)
    # --------------------------
    "english detective and mystery stories": "mystery & detective",
    "american detective and mystery stories": "mystery & detective",
    "english fantasy fiction": "fantasy",
    "american fantasy fiction": "fantasy",
    "english short stories": "short stories",
    "american short stories": "short stories",
    "english drama": "drama",
    "british and irish drama dramatic": "drama",
    "drama dramatic": "drama",
    # --------------------------
    # Fiction subtypes (added)
    # --------------------------
    "suspense fiction": "suspense",
    "horror fiction": "horror",
    "romance literature": "romance",
    "fiction classics": "classics",
    "literature and fiction": "fiction",
    "children's literature": "children's fiction",
    "graphic novels": "comics & graphic novels",
    # --------------------------
    # Young adult merging (added)
    # --------------------------
    "young adult fiction": "young adult fiction",
    "child and youth fiction": "young adult fiction",
    # --------------------------
    # Coming of age (added)
    # --------------------------
    "bildungsromans": "coming of age",
    # --------------------------
    # Historical romance - keep distinct for now (can be merged to "romance" if desired)
    # --------------------------
    "historical romance": "historical romance",
    # --------------------------
    # Misc additional cleanup / normalization artifacts (kept)
    # --------------------------
    # (these were present in the prior version; keep to avoid regressions)
    "american fiction": "american fiction",
    "british and irish fiction": "british and irish fiction",
    "continental european fiction": "continental european fiction",
}


stats = {
    "removed": 0,
    "mapped": 0,
    "unchanged": 0,
}


def normalize_subject(s):
    """Normalize a subject using mapping and simple heuristics."""
    if not isinstance(s, str):
        return None

    s_lower = s.lower().strip()

    # Heuristic removal: any subject mentioning "reading level"
    if "reading level" in s_lower:
        stats["removed"] += 1
        return None

    # Check manual mapping
    if s_lower in SUBJECT_MAPPING:
        normalized = SUBJECT_MAPPING[s_lower]
        if normalized is None:
            stats["removed"] += 1
            return None
        else:
            stats["mapped"] += 1
            return normalized
    else:
        stats["unchanged"] += 1
        return s_lower


def process_subjects(subjects):
    if not isinstance(subjects, list):
        return []

    result = []
    for subj in subjects:
        if not isinstance(subj, str):
            continue

        normalized = normalize_subject(subj)
        if normalized:
            result.append(normalized)

    # Deduplicate
    result = list(dict.fromkeys(result))
    return result


total_in = 0
total_out = 0

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

        total_in += original_count
        total_out += len(cleaned_subjects)

        outfile.write(
            json.dumps({"work_id": work_id, "subjects": cleaned_subjects}, ensure_ascii=False)
            + "\n"
        )

print(f"Cleaned {INPUT_JSONL.name}")
print("\nNormalization:")
print(f"  Removed (general, etc): {stats['removed']:,}")
print(f"  Mapped to other subjects: {stats['mapped']:,}")
print(f"  Unchanged: {stats['unchanged']:,}")
print("\nSubject counts:")
print(f"  Before: {total_in:,}")
print(f"  After: {total_out:,}")
print(f"  Removed: {total_in - total_out:,} ({(1 - total_out / total_in) * 100:.1f}%)")
print(f"\nOutput: {OUTPUT_JSONL}")
