# tests/unit/chatbot/test_base_agent.py
"""
Unit tests for BaseLangGraphAgent pure methods.

The base agent delegates all ReAct logic to LangGraph's create_react_agent,
so the testable surface is the message-building and result-extraction layer
shared by WebAgent, DocsAgent, ResponseAgent, and RetrievalAgent.

Targets:
    _convert_history(history)                       → List[BaseMessage]
    _build_messages(request, **context)             → List[BaseMessage]
    _extract_tool_calls_from_messages(messages)     → List[dict]
    _get_tool_start_status(tool_name, args)         → str
    _extract_tool_call(event)                       → Optional[dict]
    _extract_final_text(event)                      → Optional[str]
    _tokenize_for_streaming(text)                   → List[str]

All tests are synchronous and make no LLM, database, or filesystem calls.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.domain.entities import AgentRequest

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def agent():
    """
    ResponseAgent with all construction-time side effects patched out.

    ResponseAgent has no tools, making it the lightest concrete subclass.
    All methods under test live in BaseLangGraphAgent and are unaffected
    by subclass choice.
    """
    with (
        patch("app.agents.infrastructure.base_langgraph_agent.get_llm", return_value=MagicMock()),
        patch(
            "app.agents.infrastructure.base_langgraph_agent.create_react_agent",
            return_value=MagicMock(),
        ),
        patch("app.agents.infrastructure.base_langgraph_agent.append_chatbot_log"),
        patch("app.agents.infrastructure.response_agent.read_prompt", return_value="prompt"),
    ):
        from app.agents.infrastructure.response_agent import ResponseAgent

        return ResponseAgent()


@pytest.fixture
def agent_with_tools():
    """
    WebAgent fixture for tests that need tool metadata routing
    (_get_tool_start_status with a real tool list).
    """
    with (
        patch("app.agents.infrastructure.base_langgraph_agent.get_llm", return_value=MagicMock()),
        patch(
            "app.agents.infrastructure.base_langgraph_agent.create_react_agent",
            return_value=MagicMock(),
        ),
        patch("app.agents.infrastructure.base_langgraph_agent.append_chatbot_log"),
        patch("app.agents.infrastructure.web_agent.read_prompt", return_value="prompt"),
    ):
        from app.agents.infrastructure.web_agent import WebAgent

        return WebAgent()


def _make_request(user_text: str = "test query", history: list = None) -> AgentRequest:
    return AgentRequest(
        user_text=user_text,
        conversation_history=history or [],
    )


def _make_ai_message_with_tool_call(tool_name: str, args: dict, call_id: str):
    """Build a minimal AIMessage that has a tool_calls attribute."""
    msg = MagicMock(spec=AIMessage)
    msg.tool_calls = [{"id": call_id, "name": tool_name, "args": args}]
    del msg.tool_call_id
    return msg


def _make_tool_message(call_id: str, content: str):
    """Build a minimal ToolMessage (has tool_call_id and content)."""
    msg = MagicMock()
    msg.tool_call_id = call_id
    msg.content = content
    del msg.tool_calls
    return msg


# ==============================================================================
# _convert_history
# ==============================================================================


class TestConvertHistory:
    """
    _convert_history converts stored history dicts into LangChain message
    objects, keeping only the last 3 turns.
    """

    def test_empty_history_returns_empty_list(self, agent):
        assert agent._convert_history([]) == []

    def test_single_turn_produces_human_and_ai_messages(self, agent):
        result = agent._convert_history([{"u": "hello", "a": "hi there"}])
        assert len(result) == 2
        assert isinstance(result[0], HumanMessage)
        assert isinstance(result[1], AIMessage)

    def test_human_message_content_matches(self, agent):
        result = agent._convert_history([{"u": "my question", "a": "my answer"}])
        assert result[0].content == "my question"

    def test_ai_message_content_matches(self, agent):
        result = agent._convert_history([{"u": "q", "a": "the answer"}])
        assert result[1].content == "the answer"

    def test_truncates_to_last_three_turns(self, agent):
        history = [{"u": f"msg {i}", "a": f"resp {i}"} for i in range(6)]
        result = agent._convert_history(history)
        # 3 turns × 2 messages = 6 messages max
        assert len(result) == 6

    def test_last_turn_content_present(self, agent):
        history = [{"u": f"msg {i}", "a": f"resp {i}"} for i in range(5)]
        result = agent._convert_history(history)
        all_content = " ".join(m.content for m in result)
        assert "msg 4" in all_content
        assert "resp 4" in all_content

    def test_old_turns_excluded(self, agent):
        history = [{"u": f"msg {i}", "a": f"resp {i}"} for i in range(5)]
        result = agent._convert_history(history)
        all_content = " ".join(m.content for m in result)
        assert "msg 0" not in all_content
        assert "msg 1" not in all_content

    def test_turn_missing_assistant_key_handled(self, agent):
        """Turns with only 'u' and no 'a' must not raise."""
        result = agent._convert_history([{"u": "user only"}])
        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)

    def test_turn_missing_user_key_handled(self, agent):
        """Turns with only 'a' and no 'u' must not raise."""
        result = agent._convert_history([{"a": "assistant only"}])
        assert len(result) == 1
        assert isinstance(result[0], AIMessage)

    def test_exactly_three_turns_all_included(self, agent):
        history = [{"u": f"q{i}", "a": f"a{i}"} for i in range(3)]
        result = agent._convert_history(history)
        assert len(result) == 6


# ==============================================================================
# _build_messages
# ==============================================================================


class TestBuildMessages:
    """
    _build_messages assembles the full message list in the documented order:
    1. SystemMessage
    2. Context messages (from _add_context_messages)
    3. History (from _convert_history)
    4. Current query as final HumanMessage
    """

    def test_first_message_is_system(self, agent):
        request = _make_request()
        msgs = agent._build_messages(request)
        assert isinstance(msgs[0], SystemMessage)

    def test_last_message_is_human_with_query(self, agent):
        request = _make_request(user_text="my query")
        msgs = agent._build_messages(request)
        assert isinstance(msgs[-1], HumanMessage)
        assert msgs[-1].content == "my query"

    def test_history_appears_between_system_and_query(self, agent):
        request = _make_request(
            user_text="new query",
            history=[{"u": "old q", "a": "old a"}],
        )
        msgs = agent._build_messages(request)
        # system, human(old q), ai(old a), human(new query)
        assert len(msgs) == 4
        assert isinstance(msgs[1], HumanMessage)
        assert msgs[1].content == "old q"

    def test_empty_history_produces_two_messages(self, agent):
        request = _make_request()
        msgs = agent._build_messages(request)
        # system + query = 2 (ResponseAgent has no context messages)
        assert len(msgs) == 2

    def test_query_content_preserved(self, agent):
        request = _make_request(user_text="recommend dark fantasy")
        msgs = agent._build_messages(request)
        assert msgs[-1].content == "recommend dark fantasy"


# ==============================================================================
# _extract_tool_calls_from_messages
# ==============================================================================


class TestExtractToolCallsFromMessages:
    """
    _extract_tool_calls_from_messages pairs AIMessage tool_calls with their
    corresponding ToolMessage results using tool_call_id as the join key.
    """

    def test_empty_messages_returns_empty_list(self, agent):
        assert agent._extract_tool_calls_from_messages([]) == []

    def test_single_tool_call_extracted(self, agent):
        ai_msg = _make_ai_message_with_tool_call("web_search", {"query": "test"}, "id_1")
        tool_msg = _make_tool_message("id_1", "search results")
        result = agent._extract_tool_calls_from_messages([ai_msg, tool_msg])
        assert len(result) == 1
        assert result[0]["tool_name"] == "web_search"
        assert result[0]["arguments"] == {"query": "test"}
        assert result[0]["result"] == "search results"

    def test_multiple_tool_calls_extracted(self, agent):
        ai_msg1 = _make_ai_message_with_tool_call("web_search", {"query": "a"}, "id_1")
        tool_msg1 = _make_tool_message("id_1", "result a")
        ai_msg2 = _make_ai_message_with_tool_call("web_fetch", {"url": "http://x.com"}, "id_2")
        tool_msg2 = _make_tool_message("id_2", "page content")
        result = agent._extract_tool_calls_from_messages([ai_msg1, tool_msg1, ai_msg2, tool_msg2])
        assert len(result) == 2
        assert result[0]["tool_name"] == "web_search"
        assert result[1]["tool_name"] == "web_fetch"

    def test_tool_call_without_matching_result_has_empty_result(self, agent):
        ai_msg = _make_ai_message_with_tool_call("web_search", {"query": "x"}, "id_no_match")
        result = agent._extract_tool_calls_from_messages([ai_msg])
        assert len(result) == 1
        assert result[0]["result"] == ""

    def test_messages_without_tool_calls_ignored(self, agent):
        human = HumanMessage(content="hello")
        ai = AIMessage(content="hi")
        result = agent._extract_tool_calls_from_messages([human, ai])
        assert result == []


# ==============================================================================
# _get_tool_start_status
# ==============================================================================


class TestGetToolStartStatus:
    """
    _get_tool_start_status returns a status string when a tool begins executing.

    If the tool object has a metadata dict with a "status_message" key, that
    message is returned (with {arg_name} placeholders filled from args).
    Otherwise a generic "Using {tool_name}..." fallback is returned.
    """

    def test_unknown_tool_returns_generic_status(self, agent):
        status = agent._get_tool_start_status("some_tool", {})
        assert "some_tool" in status

    def test_tool_with_status_message_metadata(self, agent_with_tools):
        mock_tool = MagicMock()
        mock_tool.name = "mock_tool"
        mock_tool.metadata = {"status_message": "Running mock tool now..."}
        agent_with_tools.tools = [mock_tool]

        status = agent_with_tools._get_tool_start_status("mock_tool", {})
        assert status == "Running mock tool now..."

    def test_status_message_placeholder_filled(self, agent_with_tools):
        mock_tool = MagicMock()
        mock_tool.name = "help_read"
        mock_tool.metadata = {"status_message": "Reading {doc_name}..."}
        agent_with_tools.tools = [mock_tool]

        status = agent_with_tools._get_tool_start_status("help_read", {"doc_name": "intro.md"})
        assert status == "Reading intro.md..."

    def test_placeholder_format_failure_returns_raw_message(self, agent_with_tools):
        """If {placeholder} doesn't match any arg key, return message as-is."""
        mock_tool = MagicMock()
        mock_tool.name = "help_read"
        mock_tool.metadata = {"status_message": "Reading {missing_arg}..."}
        agent_with_tools.tools = [mock_tool]

        status = agent_with_tools._get_tool_start_status("help_read", {})
        assert "Reading" in status

    def test_tool_without_metadata_attribute_returns_fallback(self, agent_with_tools):
        mock_tool = MagicMock(spec=[])  # no attributes at all
        mock_tool.name = "bare_tool"
        agent_with_tools.tools = [mock_tool]

        status = agent_with_tools._get_tool_start_status("bare_tool", {})
        assert "bare_tool" in status


# ==============================================================================
# _extract_tool_call
# ==============================================================================


class TestExtractToolCall:
    """
    _extract_tool_call pulls the first tool call from an agent event dict.

    Events arrive during graph.astream() with the shape:
        {"agent": {"messages": [<last_message>]}}
    """

    def test_returns_none_for_empty_event(self, agent):
        assert agent._extract_tool_call({}) is None

    def test_returns_none_when_messages_empty(self, agent):
        assert agent._extract_tool_call({"agent": {"messages": []}}) is None

    def test_returns_none_when_no_tool_calls(self, agent):
        msg = AIMessage(content="no tools here")
        assert agent._extract_tool_call({"agent": {"messages": [msg]}}) is None

    def test_extracts_tool_call_from_event(self, agent):
        tool_call = {"id": "abc", "name": "web_search", "args": {"query": "test"}}
        msg = MagicMock()
        msg.tool_calls = [tool_call]
        result = agent._extract_tool_call({"agent": {"messages": [msg]}})
        assert result == tool_call

    def test_returns_first_tool_call_when_multiple(self, agent):
        call1 = {"id": "1", "name": "web_search", "args": {}}
        call2 = {"id": "2", "name": "web_fetch", "args": {}}
        msg = MagicMock()
        msg.tool_calls = [call1, call2]
        result = agent._extract_tool_call({"agent": {"messages": [msg]}})
        assert result == call1


# ==============================================================================
# _extract_final_text
# ==============================================================================


class TestExtractFinalText:
    """
    _extract_final_text retrieves the last message's content if it is
    a non-empty string longer than 10 characters.
    """

    def test_returns_none_for_empty_event(self, agent):
        assert agent._extract_final_text({}) is None

    def test_returns_none_when_messages_empty(self, agent):
        assert agent._extract_final_text({"agent": {"messages": []}}) is None

    def test_returns_none_for_short_content(self, agent):
        """Content ≤ 10 chars treated as noise and ignored."""
        msg = MagicMock()
        msg.content = "short"
        assert agent._extract_final_text({"agent": {"messages": [msg]}}) is None

    def test_returns_none_for_empty_content(self, agent):
        msg = MagicMock()
        msg.content = ""
        assert agent._extract_final_text({"agent": {"messages": [msg]}}) is None

    def test_returns_content_when_long_enough(self, agent):
        msg = MagicMock()
        msg.content = "Here is a full response with enough content."
        result = agent._extract_final_text({"agent": {"messages": [msg]}})
        assert result == "Here is a full response with enough content."

    def test_strips_whitespace(self, agent):
        msg = MagicMock()
        msg.content = "  Here is a full response with enough content.  "
        result = agent._extract_final_text({"agent": {"messages": [msg]}})
        assert result == "Here is a full response with enough content."


# ==============================================================================
# _tokenize_for_streaming
# ==============================================================================


class TestTokenizeForStreaming:
    """
    _tokenize_for_streaming splits text into word-level tokens.

    All words except the last get a trailing space so streaming renders
    correctly without double-spaces at token boundaries.
    """

    def test_single_word_no_trailing_space(self, agent):
        assert agent._tokenize_for_streaming("hello") == ["hello"]

    def test_two_words_first_has_trailing_space(self, agent):
        tokens = agent._tokenize_for_streaming("hello world")
        assert tokens == ["hello ", "world"]

    def test_last_token_never_has_trailing_space(self, agent):
        tokens = agent._tokenize_for_streaming("one two three")
        assert not tokens[-1].endswith(" ")

    def test_all_intermediate_tokens_have_trailing_space(self, agent):
        tokens = agent._tokenize_for_streaming("a b c d e")
        for token in tokens[:-1]:
            assert token.endswith(" "), f"Expected trailing space on '{token}'"

    def test_empty_string_returns_empty_list(self, agent):
        assert agent._tokenize_for_streaming("") == []

    def test_token_count_matches_word_count(self, agent):
        tokens = agent._tokenize_for_streaming("the quick brown fox jumps")
        assert len(tokens) == 5

    def test_rejoined_tokens_reproduce_original(self, agent):
        text = "one two three four"
        tokens = agent._tokenize_for_streaming(text)
        assert "".join(tokens) == text
