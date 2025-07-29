import torch
from sqlalchemy import func
from app.table_models import Book, BookSubject, Subject
from models.shared_utils import ModelStore

store = ModelStore()

BOOK_META = store.get_book_meta()
bayesian_tensor = store.get_bayesian_tensor()
_, book_ids = store.get_book_embeddings()

def get_search_results(query, subject_idxs, page, per_page, db):
    offset = page * per_page
    results = []

    if query.strip() == "":
        topk_idx = torch.topk(torch.tensor(bayesian_tensor), 1000).indices.tolist()
        topk_item_idxs = [book_ids[i] for i in topk_idx]

        filtered_meta = BOOK_META

        if subject_idxs:
            valid_books = db.query(BookSubject.item_idx).filter(
                BookSubject.subject_idx.in_(subject_idxs)
            ).group_by(BookSubject.item_idx).having(
                func.count(func.distinct(BookSubject.subject_idx)) == len(subject_idxs)
            ).all()
            valid_ids = set(b.item_idx for b in valid_books)
            filtered_meta = filtered_meta.loc[filtered_meta.index.intersection(valid_ids)]

        filtered_meta = filtered_meta.loc[filtered_meta.index.intersection(topk_item_idxs)]
        filtered_meta["__sort_idx__"] = filtered_meta.index.map(
            lambda i: topk_item_idxs.index(i) if i in topk_item_idxs else 1e9
        )
        filtered_meta = filtered_meta.sort_values("__sort_idx__").drop(columns="__sort_idx__")

        for item_idx, row in filtered_meta.iloc[offset:offset + per_page].iterrows():
            results.append({
                "item_idx": item_idx,
                "title": row["title"],
                "author": row.get("author", "Unknown"),
                "year": row.get("year"),
                "cover_id": row.get("cover_id"),
                "bayes_score": float(bayesian_tensor[item_idx]) if item_idx < len(bayesian_tensor) else None
            })

    else:
        q = db.query(Book).join(BookSubject, Book.item_idx == BookSubject.item_idx).filter(Book.title.ilike(f"%{query}%"))

        if subject_idxs:
            q = q.filter(BookSubject.subject_idx.in_(subject_idxs)).group_by(Book.item_idx).having(
                func.count(func.distinct(BookSubject.subject_idx)) == len(subject_idxs)
            )
        else:
            q = q.group_by(Book.item_idx)

        books = q.offset(offset).limit(per_page).all()

        for book in books:
            results.append({
                "item_idx": book.item_idx,
                "title": book.title,
                "author": book.author.name if book.author else "Unknown",
                "year": book.year,
                "cover_id": book.cover_id,
                "bayes_score": float(bayesian_tensor[book.item_idx]) if book.item_idx < len(bayesian_tensor) else None
            })

    return results