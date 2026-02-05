# app/agents/tools/docs_tools.py
"""
Modernized documentation/help tools with native interfaces.
"""

import json
from pathlib import Path
from typing import Optional, Callable

from langchain_core.tools import tool


class DocsTools:
    """Factory for documentation and help tools."""

    def __init__(self, docs_path: Optional[str] = None):
        if docs_path:
            self.docs_path = Path(docs_path)
        else:
            # Default: docs/help relative to project root
            base = Path(__file__).resolve().parents[3]
            self.docs_path = base / "docs" / "help"

        self.manifest_path = self.docs_path / "help_manifest.json"
        self._manifest: Optional[dict] = None

    def _load_manifest(self) -> dict[str, dict]:
        """Load and normalize help manifest."""
        if self._manifest is not None:
            return self._manifest

        try:
            with open(self.manifest_path, encoding="utf-8") as f:
                data = json.load(f)

            # Normalize keys and validate structure
            normalized = {}
            for alias, meta in (data or {}).items():
                if not isinstance(meta, dict):
                    continue

                alias_key = alias.strip().lower()
                file_path = meta.get("file", "").strip()

                if not alias_key or not file_path:
                    continue

                normalized[alias_key] = {
                    "file": file_path,
                    "title": meta.get("title", alias),
                    "description": meta.get("description", ""),
                    "keywords": meta.get("keywords", []),
                }

            self._manifest = normalized
            return self._manifest

        except Exception:
            self._manifest = {}
            return {}

    def _resolve_doc_path(self, name_or_alias: str) -> Optional[Path]:
        """Resolve document name/alias to file path."""
        manifest = self._load_manifest()
        key = name_or_alias.strip().lower()

        # Try manifest alias first
        if key in manifest:
            file_path = self.docs_path / manifest[key]["file"]
            if file_path.is_file():
                return file_path

        # Try literal filename
        if key.endswith(".md"):
            file_path = self.docs_path / key
            if file_path.is_file():
                return file_path

        return None

    def get_tools(self) -> list[Callable]:
        """Get all documentation tool definitions."""
        return [
            self._create_help_manifest_tool(),
            self._create_help_read_tool(),
        ]

    def _create_help_manifest_tool(self) -> Callable:
        """Tool to list available help documents."""

        @tool
        def help_manifest() -> dict[str, dict]:
            """
            Get the complete help documentation manifest.

            Lists all available help documents with their aliases,
            titles, descriptions, and keywords.

            Returns:
                Dictionary mapping aliases to document metadata
            """
            return self._load_manifest()

        help_manifest.status_message = "Checking available documentation..."
        return help_manifest

    def _create_help_read_tool(self) -> Callable:
        """Tool to read a specific help document."""

        @tool
        def help_read(doc_name: str) -> str:
            """
            Read a help document by alias or filename.

            Use help_manifest() first to see available documents.

            Args:
                doc_name: Document alias (e.g., 'overview') or filename

            Returns:
                Document contents as markdown text
            """
            doc_path = self._resolve_doc_path(doc_name)

            if not doc_path:
                return (
                    f"[Help] Document '{doc_name}' not found. "
                    f"Use help_manifest() to see available documents."
                )

            try:
                with open(doc_path, encoding="utf-8") as f:
                    content = f.read()
                return content if content else "[Help] Document is empty."
            except Exception as e:
                return f"[Help Error] Could not read document: {e}"

        help_read.status_message = "Reading {doc_name}..."
        return help_read

    def render_manifest_for_prompt(
        self, max_items: Optional[int] = None, include_keywords: bool = False
    ) -> str:
        """
        Render manifest as a compact string for LLM prompts.

        Args:
            max_items: Maximum number of documents to include
            include_keywords: Whether to include keyword lists

        Returns:
            Formatted string listing available documents
        """
        manifest = self._load_manifest()
        if not manifest:
            return "(no documentation available)"

        aliases = sorted(manifest.keys())
        if max_items:
            aliases = aliases[:max_items]

        lines = []
        for alias in aliases:
            meta = manifest[alias]
            title = meta.get("title", alias)
            desc = meta.get("description", "")
            keywords = meta.get("keywords", [])

            line = f"- {alias} — {title}"
            if desc:
                line += f": {desc}"
            if include_keywords and keywords:
                kw_str = ", ".join(str(k) for k in keywords)
                line += f" [keywords: {kw_str}]"

            lines.append(line)

        return "\n".join(lines)
