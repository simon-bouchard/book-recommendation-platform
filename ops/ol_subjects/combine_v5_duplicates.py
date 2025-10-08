# ops/ol_subjects/combine_v5_duplicates.py
# Reads:  data/ol_subjects/ol_subjects_cleaned_v5.jsonl
# Writes: data/ol_subjects/ol_subjects_cleaned_v5_combined.jsonl
#
# Produces one record per work_id, combining/deduplicating subjects
# (preserves first-seen subject order).

import json
from pathlib import Path
from collections import OrderedDict

ROOT = Path(__file__).resolve().parents[2]
IN_PATH = ROOT / "data" / "ol_subjects" / "ol_subjects_cleaned_v5.jsonl"
OUT_PATH = ROOT / "data" / "ol_subjects" / "ol_subjects_cleaned_v5_combined.jsonl"

def combine_subjects_v5(in_path: Path, out_path: Path):
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    # Map: work_id_str -> (list_of_subjects_preserving_order, set_of_seen_subjects)
    combined = {}
    total_rows = 0
    malformed = 0

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            total_rows += 1
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue

            work_id = rec.get("work_id")
            if work_id is None:
                malformed += 1
                continue

            # normalize work_id string for consistent keys
            work_id_s = str(work_id).strip()

            subjects = rec.get("subjects") or []
            if not isinstance(subjects, list):
                # skip or coerce non-list subject field
                try:
                    subjects = list(subjects)
                except Exception:
                    subjects = []

            if work_id_s not in combined:
                # store an ordered list and a set for de-duping
                combined[work_id_s] = {
                    "subjects_list": [],
                    "subjects_set": set()
                }

            slot = combined[work_id_s]
            for subj in subjects:
                # only handle strings
                if not isinstance(subj, str):
                    continue
                subj_s = subj.strip()
                if subj_s == "":
                    continue
                if subj_s not in slot["subjects_set"]:
                    slot["subjects_list"].append(subj_s)
                    slot["subjects_set"].add(subj_s)

    # Write out combined results
    kept = 0
    with out_path.open("w", encoding="utf-8") as out:
        for work_id_s, data in combined.items():
            rec_out = {
                "work_id": work_id_s,
                "subjects": data["subjects_list"]
            }
            out.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            kept += 1

    # Stats
    print("Input rows:", total_rows)
    print("Malformed/skipped rows:", malformed)
    print("Unique work_id after combine:", kept)
    print("Output path:", out_path)

    return {
        "input_rows": total_rows,
        "malformed": malformed,
        "unique_after": kept,
        "output_path": str(out_path)
    }

if __name__ == "__main__":
    combine_subjects_v5(IN_PATH, OUT_PATH)
