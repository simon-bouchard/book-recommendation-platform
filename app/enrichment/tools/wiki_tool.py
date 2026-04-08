# app/enrichment/wiki_tool.py
from __future__ import annotations

import html
import re
from typing import Dict, List, Optional

import requests

WIKI_API = "https://en.wikipedia.org/w/api.php"
WD_ENTITY_API = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"

# ---- Public entrypoint ------------------------------------------------------


def fetch_wiki_context_block(
    title: str,
    author: Optional[str] = None,
    target_words: int = 150,
    min_words: int = 100,
    max_words: int = 170,
    timeout_s: float = 4.0,
) -> Optional[str]:
    """
    Return a compact, deterministic '[WIKI CONTEXT]' block for a book or None if not found.

    Header lines (included if available): Title, Author, Original publication, Language, Genre
    Snippet: ~150 words (floor 100, cap 170) from the very start of the article body,
             trimmed to the nearest sentence boundary when possible.

    This function is conservative: it only returns a block when an unambiguous page is found.
    """
    page = _search_and_get_page(title, author, timeout_s=timeout_s)
    if not page:
        return None
    if page.get("disambiguation"):
        # Avoid ambiguous pages
        return None

    # Extract clean text
    text = _get_plain_text(page["pageid"], timeout_s=timeout_s) or ""
    if not text.strip():
        return None
    cleaned = _clean_wiki_text(text)

    # Build header from Wikidata if available
    header = _build_header_from_wikidata(
        page.get("qid"), fallback_title=page["title"], timeout_s=timeout_s
    )

    # Build word-window snippet from the very start of the cleaned text
    snippet = _word_window_snippet(
        cleaned, target_words=target_words, min_words=min_words, max_words=max_words
    )

    lines = ["[WIKI CONTEXT]"]
    lines.extend(_format_header(header))
    lines.append("")  # blank line before snippet
    lines.append("Snippet:")
    lines.append(f'"{snippet}"')
    return "\n".join(lines)


# ---- Wikipedia & Wikidata helpers ------------------------------------------


def _search_and_get_page(title: str, author: Optional[str], timeout_s: float) -> Optional[Dict]:
    """Search enwiki for '<title> <author> (book)' and return a resolved page dict."""
    q = title.strip()
    if author:
        q = f"{q} {author.strip()} (book)"

    params = {
        "action": "query",
        "list": "search",
        "srsearch": q,
        "srlimit": 5,
        "srprop": "",
        "format": "json",
    }
    try:
        r = requests.get(WIKI_API, params=params, timeout=timeout_s)
        r.raise_for_status()
        hits = r.json().get("query", {}).get("search", [])
    except Exception:
        return None
    if not hits:
        return None

    # Prefer exact/near-exact title matches first
    title_norm = _norm(title)
    pick = None
    for h in hits:
        if _norm(h.get("title", "")) == title_norm:
            pick = h
            break
    if not pick:
        pick = hits[0]

    # Resolve page info + pageprops (to detect disambiguation and get Wikidata QID)
    params = {
        "action": "query",
        "prop": "info|pageprops|categories",
        "titles": pick["title"],
        "inprop": "url",
        "clshow": "!hidden",
        "cllimit": 50,
        "format": "json",
    }
    try:
        r = requests.get(WIKI_API, params=params, timeout=timeout_s)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()))
    except Exception:
        return None
    if not page or "missing" in page:
        return None

    # Disambiguation detection
    disambig = False
    pp = page.get("pageprops", {})
    if "disambiguation" in pp:
        disambig = True
    cats = page.get("categories") or []
    if any("disambiguation" in (c.get("title") or "").lower() for c in cats):
        disambig = True

    qid = (pp.get("wikibase_item") or "").strip() or None
    return {
        "pageid": page.get("pageid"),
        "title": page.get("title", ""),
        "qid": qid,
        "disambiguation": disambig,
    }


def _get_plain_text(pageid: int, timeout_s: float) -> Optional[str]:
    """Fetch full plain text (no HTML) using extracts API."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "exsectionformat": "plain",
        "pageids": pageid,
        "format": "json",
    }
    try:
        r = requests.get(WIKI_API, params=params, timeout=timeout_s)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        page = pages.get(str(pageid), {})
        return page.get("extract")
    except Exception:
        return None


def _build_header_from_wikidata(
    qid: Optional[str], fallback_title: str, timeout_s: float
) -> Dict[str, str]:
    """Get Title/Author/Original publication/Language/Genre from Wikidata when possible."""
    header = {"Title": fallback_title}
    if not qid:
        return header
    try:
        r = requests.get(WD_ENTITY_API.format(qid=qid), timeout=timeout_s)
        r.raise_for_status()
        data = r.json()["entities"][qid]
    except Exception:
        return header

    labels = (data.get("labels") or {}).get("en", {}).get("value") or fallback_title
    header["Title"] = labels

    claims = data.get("claims", {})

    def one_or_many(prop: str, max_items: int = 2) -> List[str]:
        vals = []
        for snak in claims.get(prop, [])[:max_items]:
            mainsnak = snak.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            v = datavalue.get("value")
            if not v:
                continue
            # Item (e.g., author, genre, language)
            if datavalue.get("type") == "wikibase-entityid":
                q = v.get("id")
                if not q:
                    continue
                try:
                    lbl = _entity_label_en(q, timeout_s=timeout_s)
                except Exception:
                    lbl = q
                vals.append(lbl)
            # Time (e.g., publication date)
            elif datavalue.get("type") == "time":
                # time format like '+1997-00-00T00:00:00Z'
                t = v.get("time", "")
                year = _extract_year(t)
                if year:
                    vals.append(year)
        return vals

    # Author(s): P50
    authors = one_or_many("P50", max_items=2)
    if authors:
        header["Author"] = _short(", ".join(authors))

    # Original publication date: P577
    dates = one_or_many("P577", max_items=1)
    if dates:
        header["Original publication"] = dates[0]

    # Language of work: P407
    langs = one_or_many("P407", max_items=1)
    if langs:
        header["Language"] = _short(langs[0])

    # Genre: P136
    genres = one_or_many("P136", max_items=2)
    if genres:
        header["Genre"] = _short(", ".join(genres))

    return header


def _entity_label_en(qid: str, timeout_s: float) -> str:
    """Fetch an entity's English label (fall back to ID)."""
    url = WD_ENTITY_API.format(qid=qid)
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()["entities"][qid]
    return (data.get("labels") or {}).get("en", {}).get("value") or qid


# ---- Cleaning & snippet selection ------------------------------------------

_REF_RE = re.compile(r"\[\d+(?:\s*[–-]\s*\d+)?\]|(\[citation needed\])", re.IGNORECASE)
_IPA_RE = re.compile(r"\s*\/[^\/]*\/")  # basic IPA pattern between slashes
_PAREN_IPA_RE = re.compile(r"\s*\([^)]*IPA[^)]*\)", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_PUNCT_TRIM_RE = re.compile(r"\s*([,;:])")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _clean_wiki_text(text: str) -> str:
    s = html.unescape(text or "")
    # remove common citation markers / IPA / extra punctuation spacing
    s = _REF_RE.sub("", s)
    s = _PAREN_IPA_RE.sub("", s)
    s = _IPA_RE.sub("", s)
    s = _PUNCT_TRIM_RE.sub(r"\1 ", s)
    s = _WS_RE.sub(" ", s)
    return s.strip()


def _word_window_snippet(text: str, target_words: int, min_words: int, max_words: int) -> str:
    """
    Take a word-window from the very start of the article, aiming for target_words,
    bounded by [min_words, max_words], and end at a sentence boundary when possible.
    """
    words = text.split()
    if len(words) <= max_words:
        # whole text is short; keep it (bounded by min)
        chosen = " ".join(words[: max(len(words), min_words)])
        return chosen

    # Start with the first max_words words, then try to trim back to nearest sentence end <= target_words
    max_chunk = " ".join(words[:max_words])
    # Split into sentences
    sentences = _SENT_SPLIT_RE.split(max_chunk)
    out = []
    total = 0
    for sent in sentences:
        wcount = len(sent.split())
        if total + wcount > max_words:
            break
        out.append(sent)
        total += wcount
        if total >= target_words:
            break

    if not out:
        # Fallback: just cut at max_words
        return " ".join(words[:max_words]).strip()

    snippet = " ".join(out).strip()
    # Ensure minimum length; if too short, append the next sentence(s) up to max_words
    if len(snippet.split()) < min_words:
        for sent in sentences[len(out) :]:
            wcount = len(sent.split())
            if len(snippet.split()) + wcount > max_words:
                break
            snippet = (snippet + " " + sent).strip()
            if len(snippet.split()) >= min_words:
                break
    return snippet


# ---- Formatting helpers -----------------------------------------------------


def _format_header(h: Dict[str, str]) -> List[str]:
    order = ["Title", "Author", "Original publication", "Language", "Genre"]
    lines = []
    for k in order:
        v = (h.get(k) or "").strip()
        if v:
            lines.append(f"{k}: {v}")
    return lines


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _extract_year(raw_time: str) -> Optional[str]:
    # raw format: '+1997-00-00T00:00:00Z' or '+0850-00-00T...'
    m = re.match(r"^\+?(-?\d{1,6})-", raw_time or "")
    if not m:
        return None
    year = m.group(1)
    if year.startswith("0") and len(year) > 1:
        year = year.lstrip("0")
    return year or None


def _short(s: str, limit: int = 60) -> str:
    s = s.strip()
    return s if len(s) <= limit else (s[: limit - 1] + "…")
