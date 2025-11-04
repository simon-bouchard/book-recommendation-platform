# app/enrichment/prompts.py
"""
Prompt templates for enrichment.
"""
import json

SYSTEM = """You are a book metadata enrichment assistant. Your task is to analyze book information and provide structured tags following exact specifications.

**Core Principles:**
- Be specific and concrete
- Avoid generic or administrative terms
- Distinguish fiction from non-fiction carefully
- Follow tier-specific requirements exactly

**Output Format:**
Always return valid JSON with these exact fields:
- subjects: array of strings (unique noun phrases)
- tone_ids: array of integers (tone IDs from provided list)
- genre_id: single integer (genre ID from provided list)
- vibe: string (descriptive phrase, length depends on tier)

**Critical:** Count your words carefully for vibe field. Responses with incorrect word counts will be rejected."""

# ============================================================================
# TIER INSTRUCTIONS
# ============================================================================

TIER_INSTRUCTIONS = {
    "RICH": """**RICH TIER** (Full metadata available)
Generate comprehensive enrichment with high specificity:
- subjects: 6-12 specific, domain-relevant noun phrases (avoid generic terms)
- tone_ids: 1-3 tone IDs that capture the book's atmosphere
- genre_id: exactly 1 genre ID
- vibe: EXACTLY 8-12 words - a distinctive descriptive phrase

Focus on specificity and uniqueness. Make the vibe memorable and distinctive.""",
    
    "SPARSE": """**SPARSE TIER** (Limited but usable metadata)
Generate focused enrichment - balance specificity with confidence:
- subjects: 3-8 subjects (prefer specific over generic, but don't speculate)
- tone_ids: 1-3 tone IDs (only if reasonably confident)
- genre_id: exactly 1 genre ID
- vibe: EXACTLY 4-8 words if confident, empty string "" if uncertain

Only use catalog subjects if they clearly fit.""",
    
    "MINIMAL": """**MINIMAL TIER** (Limited metadata available)
Generate basic enrichment - extreme conservatism required:
- subjects: 1-3 obvious subjects only (explicitly mentioned or clear from title)
- tone_ids: 0-1 tone ID only if VERY clear, usually empty array []
- genre_id: exactly 1 genre ID
- vibe: MUST be empty string "" (NO vibe for this tier)

Do NOT infer or speculate. Only extract what's directly stated.""",
    
    "BASIC": """**BASIC TIER** (Very sparse metadata)
Genre classification only:
- subjects: 0-1 subject (only if obvious from title, otherwise empty array [])
- tone_ids: MUST be empty array []
- genre_id: exactly 1 genre ID
- vibe: MUST be empty string "" (NO vibe for this tier)

Primary goal: assign correct genre. Don't attempt to infer beyond what's explicit."""
}


# ============================================================================
# ONTOLOGY INSTRUCTIONS (updated for V2 and genre IDs)
# ============================================================================

def render_tone_instructions(tone_slugs: str, ontology_version: str = "v2") -> str:
    """Render tone selection instructions with ID mappings."""
    if ontology_version == "v2":
        return f"""**Tones (v2, 36 options) - Select 1-3 tone IDs:**
{tone_slugs}

Organized in 6 buckets to help you choose:
• PACING: slow-burn, steady, fast-paced, action-packed
• MOOD: cozy, whimsical, heartwarming, bittersweet, nostalgic, melancholic, dark, eerie, atmospheric, suspenseful, uplifting
• INTENSITY: gentle, poignant, gritty, intense, harrowing, disturbing
• HUMOR: witty, dry, satirical, ironic, absurdist, darkly-humorous
• STYLE: lyrical, evocative, ornate, spare, experimental
• ACCESSIBILITY: breezy, accessible, layered, cerebral

Select tone IDs that capture the book's FEEL and atmosphere.
Example: For a dark, fast-paced thriller → tone_ids: [11, 3, 14] (dark, fast-paced, suspenseful)"""
    else:
        return f"""**Tones (v1, 55 options) - Select 1-3 tone IDs:**
{tone_slugs}

Example: tone_ids: [1, 15, 23]"""


def render_genre_instructions(genre_id_slugs: str) -> str:
    """Render genre selection instructions with ID mappings."""
    return f"""**Genres (40 options) - Select ONE genre_id:**
{genre_id_slugs}

Key mappings for common topics:
• astrology/tarot/feng shui → religion-spirituality
• astronomy → science-nature
• self-improvement/happiness/public speaking → psychology-self-help
• ESL/grammar/spelling → language-learning
• sports/fitness/outdoor activities → lifestyle
• fashion/beauty/crafts/weddings → lifestyle
• humor/joke collections → lifestyle

**Critical:** Distinguish fiction from nonfiction carefully. 
Introductions, translations, commentaries, study guides, primers → use nonfiction genre IDs (history, biography, etc.), NOT fiction genre IDs.

Example: For a space opera → genre_id: 2
Example: For a dark thriller → genre_id: 5
Example: For a biography → genre_id: 13"""


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
    genre_id_slugs: str,
    ontology_version: str = "v2"
) -> str:
    """Build tier-specific user prompt with explicit ID requirements."""
    
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
    genre_instr = render_genre_instructions(genre_id_slugs)
    
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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{tone_instr}

{genre_instr}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Output Requirements:**
• subjects: Unique noun phrases (1-4 words each), specific and distinct
• tone_ids: Array of tone IDs from list above (integers, NOT slugs)
• genre_id: Single genre ID from list above (integer, NOT slug)
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
{"Count your words! RICH tier requires EXACTLY 8-12 words." if tier == "RICH" else ""}
{"Count your words! SPARSE tier requires EXACTLY 4-8 words (or empty if uncertain)." if tier == "SPARSE" else ""}
{"NO vibe allowed for this tier - use empty string." if tier in ("MINIMAL", "BASIC") else ""}

Return JSON:
{{"subjects": [...], "tone_ids": [11, 3, 14], "genre_id": 5, "vibe": "text or empty"}}"""
    
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
    genre_id_slugs: str,
    ontology_version: str,
    feedback: dict
) -> str:
    """
    Build retry prompt with specific feedback about what went wrong.
    
    Args:
        ... (same as build_user_prompt)
        feedback: Dict with:
            - error_type: "vibe_too_short" | "vibe_too_long" | "invalid_genre_id" | etc.
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
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  VALIDATION ERROR - RETRY NEEDED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Your previous response was rejected for this reason:
{feedback['error_msg']}

What you need to fix:
{feedback['required_changes']}

Your original response (that was rejected):
{original_json}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

**Reminder - Use IDs not slugs:**
• tone_ids: use integers like [11, 3, 14]
• genre_id: use single integer like 5

Return corrected JSON:
{{"subjects": [...], "tone_ids": [11, 3, 14], "genre_id": 5, "vibe": "corrected text or empty"}}"""
    
    return prompt
