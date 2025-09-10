# ===== Subject ID Search (3-gram TF-IDF) =====================================

import math
import unicodedata
import re
from collections import Counter, defaultdict
from typing import Tuple, List, Dict, Any
import json
from sqlalchemy.orm import Session

# Index caches (module-level; warm on first use)
_SUBJ_INDEX_BUILT = False
_SUBJ_DOCS: List[Dict[str, Any]] = []   # [{subject_idx, subject, count, norm, grams:Counter, tokens:set}]
_IDF: Dict[str, float] = {}             # 3-gram -> idf
_VEC_NORMS: Dict[int, float] = {}       # subject_idx -> ||tfidf||_2
_MAX_LOGP: float = 1.0                   # normalizer for popularity prior


def _normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)       # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _char_ngrams(s: str, n: int = 3) -> List[str]:
    s2 = f"  {s}  "  # boundaries
    return [s2[i:i+n] for i in range(max(0, len(s2)-n+1))] if s2 else []

def _tokens(s: str) -> List[str]:
    return [t for t in re.split(r"\s+", s) if t]

def _build_subject_index(db: Session):
    global _SUBJ_INDEX_BUILT, _SUBJ_DOCS, _IDF, _VEC_NORMS, _MAX_LOGP
    if _SUBJ_INDEX_BUILT:
        return

    from app.models import get_all_subject_counts  # reuse your cache
    rows = get_all_subject_counts(db)  # [{"subject_idx","subject","count"}, ...]
    if not rows:
        _SUBJ_INDEX_BUILT = True
        _SUBJ_DOCS, _IDF, _VEC_NORMS, _MAX_LOGP = [], {}, {}, 1.0
        return

    # Precompute popularity prior normalizer
    _MAX_LOGP = max(1.0, max(math.log1p(int(r.get("count", 0))) for r in rows))

    # 1) Collect docs with normalized fields & raw 3-gram counts
    docs = []
    df_counter = Counter()  # document frequency per 3-gram
    for r in rows:
        sid = int(r["subject_idx"])
        name = str(r["subject"] or "")
        cnt = int(r.get("count", 0))

        norm = _normalize_text(name)
        grams = Counter(_char_ngrams(norm, 3))
        toks = set(_tokens(norm))

        # Skip empty/degenerate subjects safely
        if not norm or not grams:
            continue

        docs.append({"subject_idx": sid, "subject": name, "count": cnt, "norm": norm, "grams": grams, "tokens": toks})
        for g in grams.keys():
            df_counter[g] += 1

    N = max(1, len(docs))
    # 2) IDF
    _IDF = {g: math.log((N + 1) / (df + 0.5)) + 1.0 for g, df in df_counter.items()}

    # 3) Store TF-IDF vector norms for subjects (for cosine)
    _VEC_NORMS = {}
    for d in docs:
        s = 0.0
        for g, tf in d["grams"].items():
            idf = _IDF.get(g, 0.0)
            s += (tf * idf) ** 2
        _VEC_NORMS[d["subject_idx"]] = math.sqrt(s) if s > 0 else 1.0

    _SUBJ_DOCS = docs
    _SUBJ_INDEX_BUILT = True


def _tfidf_cosine(query_grams: Counter, subj_grams: Counter) -> float:
    # dot
    dot = 0.0
    for g, qtf in query_grams.items():
        idf = _IDF.get(g)
        if idf is None:  # unseen gram
            continue
        stf = subj_grams.get(g)
        if stf:
            dot += (qtf * idf) * (stf * idf)
    # norms
    q_norm = math.sqrt(sum((qtf * _IDF.get(g, 0.0)) ** 2 for g, qtf in query_grams.items())) or 1.0
    return dot / q_norm  # we'll divide by doc norm downstream to save a dict lookup during loop


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / float(len(a | b))


def make_subject_id_search_tool(db: Session):
    """
    Input (JSON): {"phrases": ["light mysteries", "heist"], "top_k": 5}
    Output (JSON): [{"phrase": "...", "candidates": [{"subject_idx":..., "subject":"...", "score":0.93}, ...]}]
    """
    def _tool(arg: str = "") -> str:
        try:
            obj = json.loads(arg or "{}") if arg and arg.strip() else {}
            phrases = obj.get("phrases") or []
            top_k = int(obj.get("top_k") or 5)
            top_k = max(1, min(10, top_k))

            _build_subject_index(db)

            # Short-circuit if no index
            if not _SUBJ_DOCS:
                return json.dumps([{"phrase": p, "candidates": []} for p in phrases], ensure_ascii=False)

            out = []
            for p in phrases:
                p_norm = _normalize_text(str(p or ""))
                if not p_norm:
                    out.append({"phrase": p, "candidates": []})
                    continue

                q_grams = Counter(_char_ngrams(p_norm, 3))
                q_tokens = set(_tokens(p_norm))

                candidates = []
                for d in _SUBJ_DOCS:
                    # Cosine (we divide by doc norm here)
                    cos_num = _tfidf_cosine(q_grams, d["grams"])
                    denom = _VEC_NORMS.get(d["subject_idx"], 1.0)
                    cos = cos_num / (denom or 1.0)

                    # Token set Jaccard as a sanity gate
                    jac = _jaccard(q_tokens, d["tokens"])

                    # Gate: allow fuzzy near-matches but avoid junk
                    # - if very short queries, demand stronger overlap
                    if len(q_tokens) <= 2:
                        if jac < 0.30 and cos < 0.35:
                            continue
                    else:
                        if jac < 0.20 and cos < 0.30:
                            continue

                    # Popularity prior (gentle)
                    prior = math.log1p(int(d["count"])) / _MAX_LOGP if _MAX_LOGP > 0 else 0.0

                    # Blend score (tunable)
                    score = 0.75 * cos + 0.25 * prior

                    candidates.append({
                        "subject_idx": int(d["subject_idx"]),
                        "subject": d["subject"],
                        "score": round(float(score), 6)
                    })

                # Top-k by score
                candidates.sort(key=lambda x: x["score"], reverse=True)
                out.append({"phrase": p, "candidates": candidates[:top_k]})

            return json.dumps(out, ensure_ascii=False)

        except Exception as e:
            return json.dumps({"error": f"subject_id_search failed: {e}"})
    return _tool
