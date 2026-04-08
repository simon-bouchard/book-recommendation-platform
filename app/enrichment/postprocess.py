import re
from typing import Optional

import langcodes
import pycountry


def norm_subject(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def render_tone_slugs(tones_csv_rows):
    # tones_csv_rows: iterable of dicts with 'slug'
    items = sorted([(int(r["tone_id"]), r["slug"]) for r in tones_csv_rows])
    return ", ".join([f"{id}={slug}" for id, slug in items])


def render_genre_slugs(genres_csv_rows):
    # genres_csv_rows: iterable of dicts with 'slug'
    items = sorted([(int(r["genre_idx"]), r["slug"]) for r in genres_csv_rows])
    return ", ".join([f"{id}={slug}" for id, slug in items])


def clean_subjects(subjects):
    out = []
    seen = set()
    for s in subjects:
        ns = norm_subject(s)
        if not ns:
            continue
        if ns in seen:
            continue
        seen.add(ns)
        out.append(ns)
    return out


# --- lightweight normalization helpers ---
_MULTI_SEP = re.compile(r"[;,/|]")


def normalize_language(value: object) -> Optional[str]:
    """
    Return ISO 639-1 (alpha_2) or None.
    Accepts names ('Brazilian Portuguese'), alpha-2/3 ('pt','por'),
    or full BCP-47 tags ('pt-BR','zh-Hant-TW'). Fail-soft to None.
    Policy: if clearly multi-language -> None (avoid guessing).
    """
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    # Heuristic: bail if looks like multiple languages in one field
    if _MULTI_SEP.search(raw):
        return None

    # Try BCP-47 normalization first (handles 'eng', 'pt-BR', scripts, regions)
    try:
        std = langcodes.standardize_tag(raw)
        lang = langcodes.get(std).language  # just the language subtag
        if not lang:
            return None
        # Collapse to ISO 639-1 if possible (e.g., 'eng' -> 'en')
        if len(lang) == 2:
            return lang.lower()
        # Map alpha-3 to alpha-2 where available via pycountry
        entry = pycountry.languages.get(alpha_3=lang)
        if entry and getattr(entry, "alpha_2", None):
            return entry.alpha_2.lower()
    except Exception:
        pass

    # Fallback: direct code/name lookups
    val = raw.lower()

    # Direct alpha-2
    if len(val) == 2 and val.isalpha():
        # Ensure it's a real ISO-639-1 language
        try:
            entry = pycountry.languages.get(alpha_2=val)
            if entry:
                return val
        except Exception:
            pass

    # Alpha-3 code?
    if len(val) == 3 and val.isalpha():
        try:
            entry = pycountry.languages.get(alpha_3=val)
            if entry and getattr(entry, "alpha_2", None):
                return entry.alpha_2.lower()
        except Exception:
            pass

    # Name lookup (handles many common names)
    try:
        matches = [lang for lang in pycountry.languages if getattr(lang, "name", "").lower() == val]
        if not matches:
            # Some entries have 'common_name' or 'inverted_name'
            matches = [
                lang
                for lang in pycountry.languages
                if any(
                    getattr(lang, attr, "").lower() == val
                    for attr in ("common_name", "inverted_name")
                )
            ]
        if matches:
            entry = matches[0]
            if getattr(entry, "alpha_2", None):
                return entry.alpha_2.lower()
    except Exception:
        pass

    return None


_BARE_YEAR = re.compile(r"(?<!\d)(\d{1,4})(?!\d)")
_BCE_FLAG = re.compile(r"\b(BCE|BC)\b", re.IGNORECASE)
_CE_FLAG = re.compile(r"\b(CE|AD)\b", re.IGNORECASE)


def normalize_year(value: object) -> Optional[int]:
    """
    Parse a specific publication year from messy strings.
    - Supports BCE/BC (returns negative int).
    - If multiple years present, return the **earliest** concrete year.
    - Returns None if no unambiguous year.
    Policy change: no artificial minimum bound; upper sanity cap retained.
    """
    if value is None:
        return None

    # If we get a clean int, accept it (with BCE allowed if negative)
    if isinstance(value, int):
        y = value
        return y if y <= 2010 else None

    s = str(value)
    if not s.strip():
        return None

    # Detect BCE/CE markers once for the whole string
    bce = bool(_BCE_FLAG.search(s))
    # Extract all candidate years (1–4 digits)
    candidates = []
    for m in _BARE_YEAR.finditer(s):
        y = int(m.group(1))
        if y == 0:
            continue
        if y > 2010:
            continue
        candidates.append(-y if bce else y)

    if not candidates:
        return None

    # Earliest absolute year (more conservative for original publication)
    candidates.sort()
    return candidates[0]
