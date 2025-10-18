# app/enrichment/prompts.py
"""
Tier-aware prompts for book enrichment with explicit vibe examples.
"""
import json

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
# TIER-SPECIFIC INSTRUCTIONS (WITH VIBE EXAMPLES)
# ============================================================================

TIER_INSTRUCTIONS = {
    "RICH": """**RICH TIER** (Strong metadata available)
Generate comprehensive enrichment:
- subjects: 5-8 specific, unique noun phrases - be detailed and domain-specific
- tone_ids: 2-3 tones that capture style, pacing, and atmosphere
- genre: exactly 1 genre
- vibe: EXACTLY 8-12 words - evocative phrase capturing the book's essence

**VIBE EXAMPLES (8-12 words):**
✓ "Sweeping historical saga exploring love and loss across generations"
✓ "Dark psychological thriller unraveling secrets in a coastal town"
✓ "Lyrical meditation on memory, identity, and the immigrant experience"
✓ "Epic fantasy adventure through kingdoms torn by war and magic"

Quality focus: Specificity and uniqueness. Avoid generic subjects ("book", "story", "readers"). 
Make each subject distinct - no near-duplicates like "Greek gods" and "Greek deities".""",
    
    "SPARSE": """**SPARSE TIER** (Moderate metadata available)
Generate focused enrichment - accuracy over completeness:
- subjects: 3-5 clear, safe choices from available information
- tone_ids: 1-2 tones if confident, empty array [] if uncertain
- genre: exactly 1 genre
- vibe: EXACTLY 4-8 words if confident, empty string "" if uncertain

**VIBE EXAMPLES (4-8 words):**
✓ "Thought-provoking exploration of wartime ethics" (5 words)
✓ "Gripping detective story with noir atmosphere" (6 words)
✓ "Intimate portrait of family dynamics" (5 words)

Conservative approach: Better to omit vibe than force it. Only use catalog subjects if they clearly fit.""",
    
    "MINIMAL": """**MINIMAL TIER** (Limited metadata available)
Generate basic enrichment - extreme conservatism required:
- subjects: 1-3 obvious subjects only (explicitly mentioned or clear from title)
- tone_ids: 0-1 tone only if VERY clear, usually empty array []
- genre: exactly 1 genre
- vibe: MUST be empty string "" (NO vibe for this tier)

Do NOT infer or speculate. Only extract what's directly stated.""",
    
    "BASIC": """**BASIC TIER** (Very sparse metadata)
Genre classification only:
- subjects: 0-1 subject (only if obvious from title, otherwise empty array [])
- tone_ids: MUST be empty array []
- genre: exactly 1 genre 
- vibe: MUST be empty string "" (NO vibe for this tier)

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
# USER TEMPLATE (tier-aware with emphasis on vibe length)
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
    """Build tier-specific user prompt with explicit vibe requirements."""
    
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
    
    # Get vibe requirement text based on tier
    vibe_requirement = {
        "RICH": "vibe: EXACTLY 8-12 words (CRITICAL: count your words, vibes with <8 or >12 words will be rejected)",
        "SPARSE": "vibe: EXACTLY 4-8 words if confident, empty string \"\" if uncertain (count your words)",
        "MINIMAL": "vibe: MUST be empty string \"\" (no vibe for MINIMAL tier)",
        "BASIC": "vibe: MUST be empty string \"\" (no vibe for BASIC tier)"
    }.get(tier, "vibe: \"\"")
    
    prompt = f"""Book:
TITLE: {title}
AUTHOR: {author}
DESCRIPTION: {description if description else "(no description available)"}
{catalog_block}

{tier_instr}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{tone_instr}

{genre_instr}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Output Requirements:**
• subjects: Unique noun phrases (1-4 words each), specific and distinct
• tone_ids: Use tone IDs from list above (NOT slugs)
• genre: Use genre slug from list above
• {vibe_requirement}

**Subject Quality Rules:**
✗ Avoid: generic terms (book, story, readers, background, excellent)
✗ Avoid: near-duplicates (pick one: "Greek gods" OR "Greek deities", not both)
✗ Avoid: repeating stems (mythology, mythological)
✓ Prefer: specific, domain-relevant terms

**Handling Catalog Subjects:**
• Treat as hints, not facts - validate against description/knowledge
• Ignore if they conflict with description
• Never copy administrative tags

**VIBE LENGTH CHECK (for {tier} tier):**
{"Count your words carefully! RICH tier requires EXACTLY 8-12 words." if tier == "RICH" else ""}
{"Count your words! SPARSE tier requires EXACTLY 4-8 words (or empty if uncertain)." if tier == "SPARSE" else ""}
{"NO vibe allowed for this tier - use empty string." if tier in ("MINIMAL", "BASIC") else ""}
Words are separated on spaces (not hyphens) for vibe word count.

Return JSON:
{{"subjects": [...], "tone_ids": [...], "genre": "slug", "vibe": "text or empty"}}"""
    
    return prompt


# ============================================================================
# RETRY PROMPT (with feedback from validation failure)
# ============================================================================

def build_retry_prompt(
    title: str,
    author: str,
    description: str,
    ol_subjects: list[str],
    tier: str,
    tone_slugs: str,
    genre_slugs: str,
    ontology_version: str,
    feedback: dict
) -> str:
    """
    Build retry prompt with specific feedback about what went wrong.
    
    Args:
        ... (same as build_user_prompt)
        feedback: Dict with:
            - error_type: "vibe_too_short" | "vibe_too_long" | etc.
            - error_msg: Original validation error
            - original_response: The JSON you returned last time
            - required_changes: Specific guidance on what to fix
    """
    
    # Format catalog subjects
    if ol_subjects:
        ol_str = ", ".join(ol_subjects[:15])
        if len(ol_subjects) > 15:
            ol_str += f" (+{len(ol_subjects)-15} more)"
        catalog_block = f"CATALOG SUBJECTS: {ol_str}"
    else:
        catalog_block = "CATALOG SUBJECTS: (none)"
    
    # Get tier instruction (simplified for retry)
    tier_instr = TIER_INSTRUCTIONS.get(tier, TIER_INSTRUCTIONS["BASIC"])
    
    # Get vibe requirement
    vibe_requirement = {
        "RICH": "vibe: EXACTLY 8-12 words",
        "SPARSE": "vibe: EXACTLY 4-8 words OR empty string",
        "MINIMAL": "vibe: MUST be empty string",
        "BASIC": "vibe: MUST be empty string"
    }.get(tier, "vibe: \"\"")
    
    # Format original response
    original_json = json.dumps(feedback["original_response"], indent=2, ensure_ascii=False)
    
    # Build error-specific feedback
    error_section = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  VALIDATION ERROR - RETRY NEEDED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Your previous response was rejected for this reason:
{feedback['error_msg']}

What you need to fix:
{feedback['required_changes']}

Your original response (that was rejected):
{original_json}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    prompt = f"""Book (RETRY ATTEMPT):
TITLE: {title}
AUTHOR: {author}
DESCRIPTION: {description if description else "(no description available)"}
{catalog_block}

{error_section}

**What to do:**
1. Take your original response above
2. Fix ONLY the field(s) mentioned in the error
3. Keep everything else the same
4. Return the corrected JSON

{tier_instr}

**CRITICAL for {tier} tier:**
• {vibe_requirement}
• Count your words carefully before submitting
• Don't change fields that were correct

Return corrected JSON:
{{"subjects": [...], "tone_ids": [...], "genre": "slug", "vibe": "corrected text or empty"}}"""
    
    return prompt
