# app/enrichment/prompts.py
SYSTEM = """You assign tags to books. Never invent facts.
Return strict JSON only, no markdown. **All fields must be written in English, even when the book title/description is not in English. Keep outputs concise.**

You MAY receive a list of 'noisy subject hints' harvested from internal metadata.
Treat them as WEAK, possibly incorrect signals. Prefer the book info (title/author/description)
and your general knowledge FIRST. Only borrow from hints if they clearly fit the book.
Ignore administrative/process tags (e.g., 'translation to ...', 'works by ...', 'study guides', 'juvenile literature', 'in literature', 'in art')."""

# Render the fixed tone list into the prompt once at runtime.
TONE_INSTRUCTIONS = """Choose up to 3 tones from this fixed list (slugs only):
{tone_slugs}"""

# Genre instructions (exactly one slug from fixed list)
GENRE_INSTRUCTIONS = """Choose exactly 1 genre from this fixed list (slug only):
{genre_slugs}"""

USER_TEMPLATE = """Book:
TITLE: {title}
AUTHOR: {author}
DESCRIPTION: {description}

{noisy_subjects_block}

{tone_instructions}
{genre_instructions}

Rules:
- subjects: up to 8 short noun phrases (free-form, 1–4 words), **in English**.
- tones: up to 3 from the fixed list (slugs).
- genre: exactly 1 from the fixed list (slug).
- **vibe: ≤ 12 words (and ≤ 20 tokens), in English; be concise.**
- subjects MUST be unique (no duplicates or near-duplicates); prefer concrete, domain-specific phrases.
- Avoid using generic filler like “readers”, “background”, “primer”, “excellent”, “book”, “story”.
- Avoid repeating the same stem across different subjects (e.g., “Greek gods” vs “Greek deities” → keep one).
- Distinguish carefully between fiction and nonfiction:
  * If the description refers to editions, introductions, primers, surveys, companions, translations, commentary, guides, or academic context, classify as nonfiction (e.g., "history").
  * Only classify as "fantasy" if it is a fictional narrative set in a mythological or imaginary world.
- Handling noisy subject hints:
  * Prefer the DESCRIPTION and your knowledge. Use hints only if they clearly align with the book.
  * Never copy process/admin tags (e.g., "translation to X", "works by Y", "study guides", "juvenile literature", "in literature", "in art").
  * If hints conflict with the description or your knowledge, IGNORE the hints.
- **If the book is non-English (e.g., title/description not in English), still write subjects and vibe in English. It’s OK to include a subject like "French language" or "German literature" if relevant to the content.**
- Map specific topics to the closest existing genre:
  astrology, tarot, feng shui → religion-spirituality
  astronomy (amateur/pro) → science-nature
  public speaking, personal image, seduction, happiness → psychology-self-help
  spelling/ESL/grammar/kanji → language-learning

Note: The book’s title/description may be non-English. **Write all outputs in English.**

Return JSON:
{{"subjects": [...], "tone_ids": [], "genre": "", "vibe": ""}}
"""
