# app/enrichment/prompts.py
"""
Tier-aware prompts for book enrichment.
Different instructions and validation rules per quality tier.
"""

# ============================================================================
# SYSTEM PROMPT (unchanged)
# ============================================================================

SYSTEM = """You assign enrichment tags to books. Return strict JSON only, no markdown.

**Critical rules:**
- All outputs MUST be in English, even for non-English books
- Never invent facts - only extract what's clearly indicated
- Catalog subjects are WEAK signals from metadata - trust description/knowledge first
- Ignore administrative tags: "translation to X", "works by Y", "study guides", "juvenile literature", "in literature", "in art"
"""


# ============================================================================
# TIER-SPECIFIC INSTRUCTIONS
# ============================================================================

TIER_INSTRUCTIONS = {
    "RICH": """**RICH TIER** (Strong metadata available)
Generate comprehensive enrichment:
- subjects: 5-8 specific, unique noun phrases - be detailed and domain-specific
- tone_ids: 2-3 tones that capture style, pacing, and atmosphere
- genre: exactly 1 genre
- vibe: 8-12 word evocative phrase

Quality focus: Specificity and uniqueness. Avoid generic subjects ("book", "story", "readers"). 
Make each subject distinct - no near-duplicates like "Greek gods" and "Greek deities".""",
    
    "SPARSE": """**SPARSE TIER** (Moderate metadata available)
Generate focused enrichment - accuracy over completeness:
- subjects: 3-5 clear, safe choices from available information
- tone_ids: 1-2 tones if confident, empty array [] if uncertain
- genre: exactly 1 genre
- vibe: 4-8 words if confident, empty string "" if uncertain

Conservative approach: Better to omit than guess. Only use catalog subjects if they clearly fit.""",
    
    "MINIMAL": """**MINIMAL TIER** (Limited metadata available)
Generate basic enrichment - extreme conservatism required:
- subjects: 1-3 obvious subjects only (explicitly mentioned or clear from title)
- tone_ids: 0-1 tone only if VERY clear, usually empty array []
- genre: exactly 1 genre
- vibe: MUST be empty string ""

Do NOT infer or speculate. Only extract what's directly stated.""",
    
    "BASIC": """**BASIC TIER** (Very sparse metadata)
Genre classification only:
- subjects: 0-1 subject (only if obvious from title, otherwise empty array [])
- tone_ids: MUST be empty array []
- genre: exactly 1 genre (use "general" if unclear)
- vibe: MUST be empty string ""

Primary goal: assign correct genre. Don't attempt to infer beyond what's explicit."""
}


# ============================================================================
# ONTOLOGY INSTRUCTIONS (updated for V2)
# ============================================================================

def render_tone_instructions(tone_slugs: str, ontology_version: str = "v2") -> str:
    """Render tone selection instructions with bucket context."""
    if ontology_version == "v2":
        return f"""**Tones (v2, 36 options):**
{tone_slugs}

Organized in 6 buckets to help you choose:
• PACING: slow-burn, fast-paced, episodic, meandering
• MOOD: dark, lighthearted, melancholic, hopeful, tense, nostalgic
• INTENSITY: intense, cozy, gritty, gentle
• HUMOR: witty, satirical, absurd, dry
• STYLE: lyrical, conversational, experimental, descriptive, minimalist
• ACCESSIBILITY: accessible, challenging, dense

Select tones that capture the book's FEEL and atmosphere."""
    else:
        return f"""**Tones (v1, 55 options):**
{tone_slugs}"""


def render_genre_instructions(genre_slugs: str) -> str:
    """Render genre selection instructions with mapping examples."""
    return f"""**Genres (39 options):**
{genre_slugs}

Key mappings:
• astrology/tarot/feng shui → religion-spirituality
• astronomy → science-nature
• self-improvement/happiness/public speaking → psychology-self-help
• ESL/grammar/spelling → language-learning
• sports/fitness/outdoor activities → lifestyle
• fashion/beauty/crafts/weddings → lifestyle
• humor/joke collections → lifestyle

**Critical:** Distinguish fiction from nonfiction carefully. Introductions, translations, commentaries, 
study guides, primers → use nonfiction genre (history, biography, etc.), NOT fiction genres."""


# ============================================================================
# USER TEMPLATE (tier-aware)
# ============================================================================

def build_user_prompt(
    title: str,
    author: str,
    description: str,
    ol_subjects: list[str],
    tier: str,
    tone_slugs: str,
    genre_slugs: str,
    ontology_version: str = "v2"
) -> str:
    """Build tier-specific user prompt (balanced verbosity)."""
    
    # Format catalog subjects (limit to 15 for token efficiency)
    if ol_subjects:
        ol_str = ", ".join(ol_subjects[:15])
        if len(ol_subjects) > 15:
            ol_str += f" (+{len(ol_subjects)-15} more)"
        catalog_block = f"CATALOG SUBJECTS: {ol_str}"
    else:
        catalog_block = "CATALOG SUBJECTS: (none)"
    
    # Get tier instruction
    tier_instr = TIER_INSTRUCTIONS.get(tier, TIER_INSTRUCTIONS["BASIC"])
    
    # Build ontology sections
    tone_instr = render_tone_instructions(tone_slugs, ontology_version)
    genre_instr = render_genre_instructions(genre_slugs)
    
    prompt = f"""Book:
TITLE: {title}
AUTHOR: {author}
DESCRIPTION: {description if description else "(no description available)"}
{catalog_block}

{tier_instr}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{tone_instr}

{genre_instr}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Output Requirements:**
• subjects: Unique noun phrases (1-4 words each), specific and distinct
• tone_ids: Use tone IDs from list above (NOT slugs)
• genre: Use genre slug from list above
• vibe: Short descriptive phrase (or "" if not required)

**Subject Quality Rules:**
✗ Avoid: generic terms (book, story, readers, background, excellent)
✗ Avoid: near-duplicates (pick one: "Greek gods" OR "Greek deities", not both)
✗ Avoid: repeating stems (mythology, mythological)
✓ Prefer: specific, domain-relevant terms

**Handling Catalog Subjects:**
• Treat as hints, not facts - validate against description/knowledge
• Ignore if they conflict with description
• Never copy administrative tags

Return JSON:
{{"subjects": [...], "tone_ids": [...], "genre": "slug", "vibe": "text or empty"}}"""
    
    return prompt


# ============================================================================
# LEGACY TEMPLATE (for backward compatibility during migration)
# ============================================================================

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
- Avoid using generic filler like "readers", "background", "primer", "excellent", "book", "story".
- Avoid repeating the same stem across different subjects (e.g., "Greek gods" vs "Greek deities" → keep one).
- Distinguish carefully between fiction and nonfiction:
  * If the description refers to editions, introductions, primers, surveys, companions, translations, commentary, guides, or academic context, classify as nonfiction (e.g., "history").
  * Only classify as "fantasy" if it is a fictional narrative set in a mythological or imaginary world.
- Handling noisy subject hints:
  * Prefer the DESCRIPTION and your knowledge. Use hints only if they clearly align with the book.
  * Never copy process/admin tags (e.g., "translation to X", "works by Y", "study guides", "juvenile literature", "in literature", "in art").
  * If hints conflict with the description or your knowledge, IGNORE the hints.
- **If the book is non-English (e.g., title/description not in English), still write subjects and vibe in English. It's OK to include a subject like "French language" or "German literature" if relevant to the content.**
- Map specific topics to the closest existing genre:
  astrology, tarot, feng shui → religion-spirituality
  astronomy (amateur/pro) → science-nature
  public speaking, personal image, happiness, self-improvement → psychology-self-help
  spelling / ESL / grammar / kanji → language-learning
  sports, fitness, outdoor activities, games (excluding board/puzzles) → lifestyle
  fashion, beauty, hair, weddings, crafts, home guides → lifestyle
  humor, comic "rules", joke collections → lifestyle

Note: The book's title/description may be non-English. **Write all outputs in English.**

Return JSON:
{{"subjects": [...], "tone_ids": [], "genre": "", "vibe": ""}}
"""
