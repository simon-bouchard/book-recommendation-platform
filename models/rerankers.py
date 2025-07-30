from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from models.shared_utils import (
    ModelStore, decompose_embeddings, compute_subject_overlap, get_read_books,
    get_candidate_book_df, filter_read_books, add_book_embeddings
)

store = ModelStore()
from models.shared_utils import PAD_IDX

class Reranker(ABC):
    @abstractmethod
    def score(self, user, candidate_ids: list[int], user_emb: np.ndarray) -> pd.DataFrame:
        """
        Return a scored DataFrame of candidate books.
        """
        pass


class GBTColdReranker(Reranker):
    def score(self, user, candidate_ids, user_emb, db: Session) -> pd.DataFrame:
        BOOK_TO_SUBJ = store.get_book_to_subj()
        cold_gbt_model = store.get_cold_gbt_model()

        df = get_candidate_book_df(candidate_ids)
        df = filter_read_books(df, user.user_id, db=db)
        df = add_book_embeddings(df)

        fav_subjects_idxs = user.fav_subjects_idxs or [PAD_IDX]
        df["subject_overlap"] = df["item_idx"].apply(
            lambda idx: compute_subject_overlap(fav_subjects_idxs, BOOK_TO_SUBJ.get(idx, []))
        )

        df["country"] = user.country
        df["filled_age"] = user.filled_age
        df["age"] = user.age

        user_emb_dict = decompose_embeddings(user_emb.unsqueeze(0), "user_emb")
        for k, v in user_emb_dict.items():
            df[k] = v

        cat_cols = ["country", "filled_year", "filled_age", "main_subject"]
        cont_cols = ["age", "year", "num_pages", "subject_overlap", "book_num_ratings"]
        emb_cols = [c for c in df.columns if c.startswith("user_emb_") or c.startswith("book_emb_")]
        features = emb_cols + cont_cols + cat_cols

        for col in cat_cols:
            df[col] = df[col].astype("category")

        df["score"] = cold_gbt_model.predict(df[features])
        return df


class GBTWarmReranker(Reranker):
    def score(self, user, candidate_ids, user_emb, db: Session) -> pd.DataFrame:
        USER_META = store.get_user_meta()
        warm_gbt_model = store.get_warm_gbt_model()

        user_row = USER_META.loc[user.user_id]

        df = get_candidate_book_df(candidate_ids)
        df = filter_read_books(df, user.user_id, db=db)
        df = add_book_embeddings(df)

        user_emb_dict = decompose_embeddings(user_emb.unsqueeze(0), prefix="user_emb")
        for k, v in user_emb_dict.items():
            df[k] = v

        df["country"] = user.country
        df["filled_age"] = user.filled_age
        df["age"] = user.age
        df["user_num_ratings"] = user_row["user_num_ratings"]
        df["user_avg_rating"] = user_row["user_avg_rating"]
        df["user_rating_std"] = user_row["user_rating_std"]

        cat_cols = ["country"]
        cont_cols = [
            "age", "year", "num_pages",
            "book_num_ratings", "book_rating_std",
            "user_num_ratings", "user_rating_std", "user_avg_rating"
        ]
        emb_cols = [c for c in df.columns if c.startswith("user_emb_") or c.startswith("book_emb_")]
        features = cont_cols + cat_cols + emb_cols

        for col in cat_cols:
            df[col] = df[col].astype("category")
        for col in cont_cols:
            df[col] = df[col].astype(np.float32)

        df["score"] = warm_gbt_model.predict(df[features])
        return df