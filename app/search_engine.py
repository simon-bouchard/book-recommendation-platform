import torch
from sqlalchemy import func
from app.table_models import Book, BookSubject, Subject
from models.shared_utils import ModelStore

# --- Strategy engines ---

class _BaseSearchStrategy:
    def __init__(self, BOOK_META, bayesian_tensor, book_ids, item_idx_to_row):
        self.BOOK_META = BOOK_META
        self.bayesian_tensor = bayesian_tensor
        self.book_ids = book_ids
        self.item_idx_to_row = item_idx_to_row

    def _bayes(self, item_idx: int) -> float:
        row = self.item_idx_to_row.get(int(item_idx))
        return float(self.bayesian_tensor[row]) if row is not None else float("-inf")

    def _paginate_df(self, df, offset, per_page):
        slice_df = df.iloc[offset:offset + per_page + 1]
        has_next = len(slice_df) > per_page
        if has_next:
            slice_df = slice_df.iloc[:per_page]
        return slice_df, has_next

    def _row_to_result(self, item_idx, row):
        return {
            "item_idx": item_idx,
            "title": row["title"],
            "author": row.get("author", "Unknown"),
            "year": row.get("year"),
            "cover_id": row.get("cover_id"),
            "isbn": row.get("isbn"),
            "bayes_score": self._bayes(item_idx),
        }

    def run(self, *args, **kwargs):
        raise NotImplementedError

class _SubjectsNoQueryStrategy(_BaseSearchStrategy):
    def run(self, subject_idxs, page, per_page, db):
        offset = page * per_page

        valid_books = (
            db.query(BookSubject.item_idx)
              .filter(BookSubject.subject_idx.in_(subject_idxs))
              .group_by(BookSubject.item_idx)
              .all()
        )
        valid_ids = {b.item_idx for b in valid_books}

        # preserve the pre-sorted global order: use .isin() mask, not .intersection()
        df = self.BOOK_META[self.BOOK_META.index.isin(valid_ids)]
        if df.empty:
            return [], False

        slice_df, has_next = self._paginate_df(df, offset, per_page)
        results = [self._row_to_result(i, r) for i, r in slice_df.iterrows()]
        return results, has_next

# app/search_engine.py
class _NoQueryGlobalStrategy(_BaseSearchStrategy):
    def run(self, page, per_page):
        offset = page * per_page
        slice_df = self.BOOK_META.iloc[offset:offset + per_page + 1]
        has_next = len(slice_df) > per_page
        if has_next:
            slice_df = slice_df.iloc[:per_page]
        results = [self._row_to_result(i, r) for i, r in slice_df.iterrows()]
        return results, has_next

class _QueryStrategy(_BaseSearchStrategy):
    # query path (title LIKE), keep your current SQL pagination + group_by/having
    def run(self, query, subject_idxs, page, per_page, db):
        offset = page * per_page

        q = (
            db.query(Book)
              .join(BookSubject, Book.item_idx == BookSubject.item_idx)
              .filter(Book.title.ilike(f"%{query}%"))
        )

        if subject_idxs:
            q = (q.filter(BookSubject.subject_idx.in_(subject_idxs))
                   .group_by(Book.item_idx)
                   .having(func.count(func.distinct(BookSubject.subject_idx)) == len(subject_idxs)))
        else:
            q = q.group_by(Book.item_idx)

        books = q.offset(offset).limit(per_page + 1).all()
        has_next = len(books) > per_page
        if has_next:
            books = books[:per_page]

        results = [{
            "item_idx": b.item_idx,
            "title": b.title,
            "author": b.author.name if b.author else "Unknown",
            "year": b.year,
            "cover_id": b.cover_id,
            "isbn": b.isbn,
            "bayes_score": self._bayes(b.item_idx),
        } for b in books]

        return results, has_next


def _get_strategy(query: str, subject_idxs, BOOK_META, bayesian_tensor, book_ids, item_idx_to_row):
    if query.strip() == "":
        if subject_idxs:
            return _SubjectsNoQueryStrategy(BOOK_META, bayesian_tensor, book_ids, item_idx_to_row)
        return _NoQueryGlobalStrategy(BOOK_META, bayesian_tensor, book_ids, item_idx_to_row)
    return _QueryStrategy(BOOK_META, bayesian_tensor, book_ids, item_idx_to_row)

def get_search_results(query, subject_idxs, page, per_page, db):
    store = ModelStore()

    BOOK_META = store.get_book_meta()
    bayesian_tensor = store.get_bayesian_tensor()
    _, book_ids = store.get_book_embeddings()
    item_idx_to_row = store.get_item_idx_to_row()  

    strat = _get_strategy(query, subject_idxs, BOOK_META, bayesian_tensor, book_ids, item_idx_to_row)
    if isinstance(strat, _SubjectsNoQueryStrategy):
        return strat.run(subject_idxs, page, per_page, db)
    if isinstance(strat, _NoQueryGlobalStrategy):
        return strat.run(page, per_page)
    # _QueryStrategy
    return strat.run(query, subject_idxs, page, per_page, db)
