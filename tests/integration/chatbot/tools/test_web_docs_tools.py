# tests/integration/chatbot/tools/test_docs_tools.py
"""
Integration tests for documentation tools using real help files.
"""

from app.agents.tools.docs_tools import DocsTools


class TestHelpManifest:
    """Tests for help_manifest tool."""

    def test_returns_manifest_dict(self):
        """
        Verify help_manifest returns dictionary of available documents.
        """
        docs_tools = DocsTools()
        help_manifest = docs_tools._create_help_manifest_tool()

        result = help_manifest.invoke({})

        assert isinstance(result, dict)

    def test_manifest_has_expected_structure(self):
        """
        Verify manifest entries have required fields.
        """
        docs_tools = DocsTools()
        help_manifest = docs_tools._create_help_manifest_tool()

        result = help_manifest.invoke({})

        if result:
            first_key = next(iter(result))
            entry = result[first_key]
            assert "file" in entry
            assert "title" in entry


class TestHelpRead:
    """Tests for help_read tool."""

    def test_reads_document_by_alias(self):
        """
        Verify help_read returns document content for valid alias.
        """
        docs_tools = DocsTools()
        help_manifest = docs_tools._create_help_manifest_tool()
        help_read = docs_tools._create_help_read_tool()

        manifest = help_manifest.invoke({})

        if manifest:
            first_alias = next(iter(manifest))
            result = help_read.invoke({"doc_name": first_alias})

            assert isinstance(result, str)
            assert len(result) > 0
            assert not result.startswith("[Help] Document")

    def test_returns_error_for_invalid_document(self):
        """
        Verify help_read returns error for non-existent document.
        """
        docs_tools = DocsTools()
        help_read = docs_tools._create_help_read_tool()

        result = help_read.invoke({"doc_name": "nonexistent_doc_12345"})

        assert isinstance(result, str)
        assert "[Help] Document" in result
        assert "not found" in result
