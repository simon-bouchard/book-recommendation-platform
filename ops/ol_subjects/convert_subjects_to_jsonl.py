# ops/ol_subjects/convert_subjects_to_jsonl.py
import json
from pathlib import Path

import pandas as pd

# Paths
ROOT = Path(__file__).resolve().parents[2]
INPUT_PKL = ROOT / "data" / "ol_subjects" / "books_with_genres.pkl"
OUTPUT_JSONL = ROOT / "data" / "ol_subjects" / "ol_subjects.jsonl"

# Load
df = pd.read_pickle(INPUT_PKL)

# Convert
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        work_id = row.get("work_id")
        subjects = row.get("subjects")

        if subjects is None or (isinstance(subjects, float) and pd.isna(subjects)):
            subjects = []
        elif not isinstance(subjects, list):
            subjects = []

        record = {"work_id": work_id, "subjects": subjects}

        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Converted {len(df):,} books to {OUTPUT_JSONL}")
