# tests/unit/recsys/test_planner_parsing.py
"""
Unit tests for PlannerAgent pure methods.

Targets:
    _parse_strategy_response(text, profile_data) → PlannerStrategy
    _build_prompt(query, available_tools, profile_data) → str

All tests are synchronous and make no LLM or I/O calls.
"""

import json

import pytest

from app.agents.domain.recsys_schemas import PlannerStrategy

# ==============================================================================
# Helpers
# ==============================================================================


def _valid_json(**overrides) -> str:
    """Return a minimal valid strategy JSON string, with optional field overrides."""
    base = {
        "recommended_tools": ["als_recs"],
        "fallback_tools": ["popular_books"],
        "reasoning": "ALS is best for this warm user.",
    }
    base.update(overrides)
    return json.dumps(base)


# ==============================================================================
# _parse_strategy_response
# ==============================================================================


class TestParseStrategyResponse:
    """
    _parse_strategy_response parses raw LLM output into a PlannerStrategy.

    The method must handle:
    - Plain JSON responses
    - JSON wrapped in ```json ... ``` fences
    - JSON wrapped in plain ``` ... ``` fences
    - Missing required fields → KeyError
    - Invalid JSON → json.JSONDecodeError
    - Profile data attachment (uses pre-fetched value, ignores LLM output)
    - Optional negative_constraints field
    """

    def test_parses_plain_json(self, planner_agent):
        result = planner_agent._parse_strategy_response(_valid_json(), profile_data=None)
        assert isinstance(result, PlannerStrategy)
        assert result.recommended_tools == ["als_recs"]
        assert result.fallback_tools == ["popular_books"]
        assert result.reasoning == "ALS is best for this warm user."

    def test_strips_json_fence(self, planner_agent):
        fenced = f"```json\n{_valid_json()}\n```"
        result = planner_agent._parse_strategy_response(fenced, profile_data=None)
        assert result.recommended_tools == ["als_recs"]

    def test_strips_plain_fence(self, planner_agent):
        fenced = f"```\n{_valid_json()}\n```"
        result = planner_agent._parse_strategy_response(fenced, profile_data=None)
        assert result.recommended_tools == ["als_recs"]

    def test_attaches_profile_data(self, planner_agent):
        profile = {"favorite_subjects": [{"subject": "SF", "subject_idx": 12}]}
        result = planner_agent._parse_strategy_response(_valid_json(), profile_data=profile)
        assert result.profile_data == profile

    def test_none_profile_data_attached(self, planner_agent):
        result = planner_agent._parse_strategy_response(_valid_json(), profile_data=None)
        assert result.profile_data is None

    def test_negative_constraints_parsed_when_present(self, planner_agent):
        text = _valid_json(negative_constraints=["vampires", "romance"])
        result = planner_agent._parse_strategy_response(text, profile_data=None)
        assert result.negative_constraints == ["vampires", "romance"]

    def test_negative_constraints_none_when_absent(self, planner_agent):
        result = planner_agent._parse_strategy_response(_valid_json(), profile_data=None)
        assert result.negative_constraints is None

    def test_missing_recommended_tools_raises_key_error(self, planner_agent):
        data = {"fallback_tools": ["popular_books"], "reasoning": "reason"}
        with pytest.raises(KeyError):
            planner_agent._parse_strategy_response(json.dumps(data), profile_data=None)

    def test_missing_fallback_tools_raises_key_error(self, planner_agent):
        data = {"recommended_tools": ["als_recs"], "reasoning": "reason"}
        with pytest.raises(KeyError):
            planner_agent._parse_strategy_response(json.dumps(data), profile_data=None)

    def test_missing_reasoning_raises_key_error(self, planner_agent):
        data = {"recommended_tools": ["als_recs"], "fallback_tools": ["popular_books"]}
        with pytest.raises(KeyError):
            planner_agent._parse_strategy_response(json.dumps(data), profile_data=None)

    def test_invalid_json_raises_json_decode_error(self, planner_agent):
        with pytest.raises(json.JSONDecodeError):
            planner_agent._parse_strategy_response("not json at all", profile_data=None)

    def test_empty_string_raises_json_decode_error(self, planner_agent):
        with pytest.raises(json.JSONDecodeError):
            planner_agent._parse_strategy_response("", profile_data=None)

    def test_multiple_tools_parsed(self, planner_agent):
        text = _valid_json(
            recommended_tools=["als_recs", "book_semantic_search"],
            fallback_tools=["popular_books", "subject_id_search"],
        )
        result = planner_agent._parse_strategy_response(text, profile_data=None)
        assert result.recommended_tools == ["als_recs", "book_semantic_search"]
        assert result.fallback_tools == ["popular_books", "subject_id_search"]

    def test_whitespace_around_json_handled(self, planner_agent):
        padded = f"\n   {_valid_json()}   \n"
        result = planner_agent._parse_strategy_response(padded, profile_data=None)
        assert result.recommended_tools == ["als_recs"]


# ==============================================================================
# _build_prompt
# ==============================================================================


class TestBuildPrompt:
    """
    _build_prompt assembles the full prompt sent to the LLM.

    Key structural requirements:
    - Contains the query
    - Lists available tools
    - Includes user rating count and ALS availability
    - Includes profile data section when provided, omits it when not
    """

    def test_contains_query(self, planner_agent):
        prompt = planner_agent._build_prompt(
            query="dark atmospheric mystery",
            available_tools=["book_semantic_search"],
            profile_data=None,
        )
        assert "dark atmospheric mystery" in prompt

    def test_lists_available_tools(self, planner_agent):
        tools = ["als_recs", "book_semantic_search", "popular_books"]
        prompt = planner_agent._build_prompt(
            query="recommend something",
            available_tools=tools,
            profile_data=None,
        )
        for tool in tools:
            assert tool in prompt

    def test_includes_user_rating_count(self, planner_agent):
        # planner_agent fixture has user_num_ratings=5
        prompt = planner_agent._build_prompt(
            query="recommend something",
            available_tools=["popular_books"],
            profile_data=None,
        )
        assert "5" in prompt

    def test_includes_als_availability_flag(self, planner_agent):
        # planner_agent fixture has has_als_recs_available=False
        prompt = planner_agent._build_prompt(
            query="q",
            available_tools=["popular_books"],
            profile_data=None,
        )
        assert "False" in prompt

    def test_no_profile_section_when_none(self, planner_agent):
        prompt = planner_agent._build_prompt(
            query="q",
            available_tools=["popular_books"],
            profile_data=None,
        )
        # _build_prompt writes this literal string when profile_data is falsy
        assert "None (no profile available)" in prompt

    def test_profile_subjects_appear_when_provided(self, planner_agent):
        profile = {
            "favorite_subjects": [{"subject": "Science Fiction", "subject_idx": 12}],
            "recent_interactions": [],
        }
        prompt = planner_agent._build_prompt(
            query="q",
            available_tools=["subject_hybrid_pool"],
            profile_data=profile,
        )
        assert "Science Fiction" in prompt

    def test_profile_subject_ids_appear_when_provided(self, planner_agent):
        profile = {
            "favorite_subjects": [{"subject": "Mystery", "subject_idx": 42}],
            "recent_interactions": [],
        }
        prompt = planner_agent._build_prompt(
            query="q",
            available_tools=["subject_hybrid_pool"],
            profile_data=profile,
        )
        assert "42" in prompt

    def test_recent_interactions_appear_when_provided(self, planner_agent):
        profile = {
            "favorite_subjects": [],
            "recent_interactions": [{"title": "Foundation", "rating": 5}],
        }
        prompt = planner_agent._build_prompt(
            query="q",
            available_tools=["als_recs"],
            profile_data=profile,
        )
        assert "Foundation" in prompt

    def test_returns_string(self, planner_agent):
        result = planner_agent._build_prompt(
            query="q",
            available_tools=["popular_books"],
            profile_data=None,
        )
        assert isinstance(result, str)
        assert len(result) > 0
