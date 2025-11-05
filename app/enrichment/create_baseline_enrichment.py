"""
Create baseline enrichment without LLM calls.
Uses raw metadata: title, author, description, OL subjects.

Output format matches enrichment JSONL structure but without LLM fields.
"""
import json
import sys
from pathlib import Path
from datetime import datetime
from sqlalchemy.orm import Session
from tqdm import tqdm

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.database import SessionLocal
from app.table_models import Book, Author, OLSubject, BookOLSubject


def fetch_book_baseline_data(db: Session, item_idx: int) -> dict:
    """
    Fetch raw book metadata for baseline (no LLM).
    
    Returns dict with:
    - title, author, description (all raw)
    - ol_subjects (raw OL subject strings)
    """
    result = db.query(Book, Author).outerjoin(
        Author, Book.author_idx == Author.author_idx
    ).filter(Book.item_idx == item_idx).first()
    
    if not result:
        return None
    
    book, author = result
    
    # Fetch OL subjects
    ol_subjects_query = db.query(OLSubject.subject).join(
        BookOLSubject, OLSubject.ol_subject_idx == BookOLSubject.ol_subject_idx
    ).filter(BookOLSubject.item_idx == item_idx).all()
    
    ol_subjects = [subj for (subj,) in ol_subjects_query if subj]
    
    return {
        "item_idx": int(book.item_idx),
        "title": book.title or "",
        "author": author.name if author else "",
        "description": book.description or "",
        "ol_subjects": ol_subjects,
    }


def create_baseline_dict(book_data: dict) -> dict:
    """
    Create baseline enrichment dict without LLM fields.
    
    Structure:
    - book_id: item_idx
    - title, author, description: raw metadata
    - subjects: raw OL subjects (no LLM processing)
    - tags_version: "baseline"
    
    Omits: vibe, genre, tone_ids (LLM-only fields)
    """
    return {
        "book_id": book_data["item_idx"],
        "title": book_data["title"],
        "author": book_data["author"],
        "description": book_data["description"],
        "subjects": book_data["ol_subjects"],
        "tags_version": "baseline",
        "enrichment_quality": "raw",
        "metadata": {
            "source": "openlibrary",
            "created_at": datetime.utcnow().isoformat(),
            "has_description": bool(book_data["description"]),
            "num_ol_subjects": len(book_data["ol_subjects"])
        }
    }


def main(output_path: str, limit: int = None):
    """
    Generate baseline enrichment JSONL for all books.
    
    Args:
        output_path: Where to write JSONL
        limit: Optional limit for testing (None = all books)
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("BASELINE ENRICHMENT GENERATOR")
    print(f"{'='*80}")
    print(f"Output: {output_file}")
    if limit:
        print(f"Limit: {limit} books (testing mode)")
    print(f"{'='*80}\n")
    
    with SessionLocal() as db:
        # Query all books with titles
        query = (
            db.query(Book.item_idx)
            .filter(Book.title.isnot(None))
            .order_by(Book.item_idx)
        )
        
        if limit:
            query = query.limit(limit)
        
        item_indices = [idx for (idx,) in query.all()]
        total = len(item_indices)
        
        print(f"Found {total:,} books to process\n")
        
        # Process and write JSONL
        written = 0
        skipped = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item_idx in tqdm(item_indices, desc="Creating baseline"):
                book_data = fetch_book_baseline_data(db, item_idx)
                
                if not book_data:
                    skipped += 1
                    continue
                
                # Skip books with no content (no title/author/desc/subjects)
                if not any([
                    book_data["title"],
                    book_data["author"],
                    book_data["description"],
                    book_data["ol_subjects"]
                ]):
                    skipped += 1
                    continue
                
                baseline_dict = create_baseline_dict(book_data)
                f.write(json.dumps(baseline_dict, ensure_ascii=False) + "\n")
                written += 1
        
        print(f"\n{'='*80}")
        print(f"✅ Baseline enrichment complete!")
        print(f"Written: {written:,} books")
        print(f"Skipped: {skipped:,} books (insufficient data)")
        print(f"Output: {output_file}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate baseline enrichment JSONL")
    parser.add_argument(
        "--output",
        default="data/baseline_enrichment.jsonl",
        help="Output JSONL path"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of books (for testing)"
    )
    
    args = parser.parse_args()
    main(args.output, args.limit)
