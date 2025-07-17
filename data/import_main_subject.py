# scripts/add_main_subjects.py

import sys
import os
from collections import Counter, defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import text, update
from app.database import SessionLocal
from app.table_models import Book, Subject, BookSubject

EXCLUDED = {'Fiction', 'General'}

def main():
    db = SessionLocal()

    try:
        # Step 1: Add column if not exists
        try:
            db.execute(text("ALTER TABLE books ADD COLUMN main_subject TEXT"))
            db.commit()
            print("✅ Added 'main_subject' column to books table")
        except Exception as e:
            print("ℹ️ Skipping column creation (already exists?)")

        # Step 2: Load all book → subject mappings
        print("📦 Loading subjects per book...")
        book_to_subjects = defaultdict(list)
        subject_counts = Counter()

        # Join: BookSubject → Subject
        results = (
            db.query(BookSubject.item_idx, Subject.subject)
            .join(Subject, BookSubject.subject_idx == Subject.subject_idx)
            .all()
        )

        for item_idx, subject in results:
            book_to_subjects[item_idx].append(subject)
            if subject not in EXCLUDED:
                subject_counts[subject] += 1

        print(f"📊 {len(subject_counts)} meaningful subjects counted")

        # Step 3: Assign most frequent subject per book
        updates = 0
        for item_idx, subjects in book_to_subjects.items():
            filtered = [s for s in subjects if s not in EXCLUDED and s in subject_counts]
            if not filtered:
                main_subj = "[NO_SUBJECT]"
            else:
                main_subj = max(filtered, key=lambda s: subject_counts[s])

            db.execute(
                update(Book)
                .where(Book.item_idx == item_idx)
                .values(main_subject=main_subj)
            )
            updates += 1

        db.commit()
        print(f"✅ Assigned main_subject to {updates} books")

    except Exception as e:
        db.rollback()
        print(f"❌ Error: {e}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
