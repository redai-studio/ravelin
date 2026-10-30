"""
This module contains MarkdownDocumentCleaner, a preprocessing component to clean and normalize markdown content.
"""

import logging
import re
from typing import List, Optional, Dict, Any

from haystack import component
from haystack.dataclasses import Document

logger = logging.getLogger(__name__)


@component
class MarkdownDocumentCleaner:
    """
    A preprocessing component to clean and normalize markdown content. This component can perform the following
    operations:

    - Fix heading formatting by ensuring a space after the # symbol
    - Fix list item formatting by ensuring proper spacing
    - Normalize line breaks for markdown elements
    - Convert typographic quotes to standard ASCII quotes
    - Strip HTML comments from the markdown
    - Remove consecutive duplicate lines
    - Preserve code blocks from modifications
    - Preserve LaTeX math expressions from modifications
    - Apply custom regex replacement rules

    Each operation can be toggled on or off using the corresponding parameters.

    :param fix_headings: Ensure headings have space after #
    :param fix_lists: Ensure list items have proper spacing
    :param fix_line_breaks: Normalize line breaks for markdown elements
    :param normalize_quotes: Convert typographic quotes to standard ASCII quotes
    :param remove_html_comments: Strip HTML comments from the markdown
    :param remove_duplicate_lines: Remove consecutive duplicate lines
    :param preserve_code_blocks: Protect code blocks from modifications
    :param preserve_latex: Protect LaTeX math expressions from modifications
    :param custom_rules: List of custom regex replacement rules to apply
        Each rule should be a dict with 'pattern', 'replacement', and optional 'flags' keys
    """

    def __init__(
        self,
        fix_headings: bool = True,
        fix_lists: bool = True,
        fix_line_breaks: bool = True,
        normalize_quotes: bool = True,
        remove_html_comments: bool = True,
        remove_duplicate_lines: bool = True,
        preserve_code_blocks: bool = True,
        preserve_latex: bool = True,
        custom_rules: Optional[List[Dict[str, Any]]] = None,
    ):
        self.fix_headings = fix_headings
        self.fix_lists = fix_lists
        self.fix_line_breaks = fix_line_breaks
        self.normalize_quotes = normalize_quotes
        self.remove_html_comments = remove_html_comments
        self.remove_duplicate_lines = remove_duplicate_lines
        self.preserve_code_blocks = preserve_code_blocks
        self.preserve_latex = preserve_latex
        self.custom_rules = custom_rules or []

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Clean and normalize markdown content in the provided documents.

        :param documents: List of Documents with markdown content
        :returns: A dictionary containing the cleaned documents under the 'documents' key
        """
        for doc in documents:
            original = doc.content
            doc.content = self._clean_content(doc.content)

            if original != doc.content:
                logger.debug("Document %s: Content was modified during cleaning", doc.id)

        return {"documents": documents}

    def _clean_content(self, markdown_text: Optional[str]) -> str:
        """
        Clean and normalize markdown content while preserving special elements.

        :param markdown_text: Raw markdown content
        :returns: Cleaned markdown content
        """
        if markdown_text is None:
            return ""

        # Convert to string if not already
        if not isinstance(markdown_text, str):
            markdown_text = str(markdown_text)

        # Store special content that should be preserved
        special_content = []
        placeholder_pattern = "PRESERVED_CONTENT_{}"

        # Function to store special content and return a placeholder
        def store_content(match):
            special_content.append(match.group(0))
            return placeholder_pattern.format(len(special_content) - 1)

        # Create a new text with preserved content replaced by placeholders
        processed_text = markdown_text

        # Preserve code blocks if enabled
        if self.preserve_code_blocks:
            # Fenced code blocks (```code```)
            processed_text = re.sub(r"```[\s\S]*?```", store_content, processed_text)

            # Inline code blocks (`code`)
            processed_text = re.sub(r"`[^`\n]+`", store_content, processed_text)

            # Indented code blocks (4+ spaces or 1+ tab at start of line)
            processed_text = re.sub(r"(?m)^((?: {4}|\t).*(?:\n(?: {4}|\t).*)*)", store_content, processed_text)

        # Preserve LaTeX if enabled
        if self.preserve_latex:
            # Block/display math with $$
            processed_text = re.sub(r"\$\$[\s\S]*?\$\$", store_content, processed_text)

            # LaTeX environments
            processed_text = re.sub(r"\\begin\{[^}]*\}[\s\S]*?\\end\{[^}]*\}", store_content, processed_text)

            # Inline math with single $
            # More careful pattern to avoid capturing currency symbols
            processed_text = re.sub(
                r"(?<![\\$a-zA-Z0-9])\$(?=\S)(?![\s$])[\s\S]*?[^\\]\$(?![a-zA-Z0-9])", store_content, processed_text
            )

            # Inline math with \( \)
            processed_text = re.sub(r"\\\([\s\S]*?\\\)", store_content, processed_text)

        # Apply cleaning operations on the text with preserved content

        # Fix line breaks
        if self.fix_line_breaks:
            processed_text = re.sub(r"(\n\s*\n\s*)+", "\n\n", processed_text)

        # Fix headings format
        if self.fix_headings:
            processed_text = re.sub(r"(#+)([^\s#])", r"\1 \2", processed_text)

        # Fix list items formatting
        if self.fix_lists:
            processed_text = re.sub(r"(\n\s*)-(\S)", r"\1- \2", processed_text)
            processed_text = re.sub(r"(\n\s*)\*(\S)", r"\1* \2", processed_text)
            processed_text = re.sub(r"(\n\s*)(\d+)\.(\S)", r"\1\2. \3", processed_text)
            # Fix task lists
            processed_text = re.sub(r"- \[([\sxX])\](\S)", r"- [\1] \2", processed_text)

        # Remove HTML comments
        if self.remove_html_comments:
            processed_text = re.sub(r"<!--[\s\S]*?-->", "", processed_text)

        # Normalize quotation marks and apostrophes
        if self.normalize_quotes:
            # Create translation mapping directly
            trans_dict = {
                # Double quotes to straight double quote (")
                ord("\u201c"): ord('"'),  # LEFT DOUBLE QUOTATION MARK
                ord("\u201d"): ord('"'),  # RIGHT DOUBLE QUOTATION MARK
                ord("\u201e"): ord('"'),  # DOUBLE LOW-9 QUOTATION MARK
                ord("\u2033"): ord('"'),  # DOUBLE PRIME
                ord("\u275d"): ord('"'),  # HEAVY DOUBLE TURNED COMMA QUOTATION MARK
                ord("\u275e"): ord('"'),  # HEAVY DOUBLE COMMA QUOTATION MARK
                ord("\u301d"): ord('"'),  # REVERSED DOUBLE PRIME QUOTATION MARK
                ord("\u301e"): ord('"'),  # DOUBLE PRIME QUOTATION MARK
                # Single quotes to straight single quote (')
                ord("\u2018"): ord("'"),  # LEFT SINGLE QUOTATION MARK
                ord("\u2019"): ord("'"),  # RIGHT SINGLE QUOTATION MARK
                ord("\u201a"): ord("'"),  # SINGLE LOW-9 QUOTATION MARK
                ord("\u275b"): ord("'"),  # HEAVY SINGLE TURNED COMMA QUOTATION MARK
                ord("\u275c"): ord("'"),  # HEAVY SINGLE COMMA QUOTATION MARK
                # Apostrophe variants to straight apostrophe (')
                ord("\u02b9"): ord("'"),  # MODIFIER LETTER PRIME
                ord("\u02bb"): ord("'"),  # MODIFIER LETTER TURNED COMMA
                ord("\u02bc"): ord("'"),  # MODIFIER LETTER APOSTROPHE
                ord("\u02be"): ord("'"),  # MODIFIER LETTER RIGHT HALF RING
                ord("\u02c8"): ord("'"),  # MODIFIER LETTER VERTICAL LINE
                ord("\u02ee"): ord("'"),  # MODIFIER LETTER DOUBLE APOSTROPHE
                ord("\u0301"): ord("'"),  # COMBINING ACUTE ACCENT
                ord("\u0313"): ord("'"),  # COMBINING COMMA ABOVE
                ord("\u0315"): ord("'"),  # COMBINING COMMA ABOVE RIGHT
                ord("\u055a"): ord("'"),  # ARMENIAN APOSTROPHE
                ord("\u05f3"): ord("'"),  # HEBREW PUNCTUATION GERESH
                ord("\u07f4"): ord("'"),  # NKO HIGH TONE APOSTROPHE
                ord("\u07f5"): ord("'"),  # NKO LOW TONE APOSTROPHE
                ord("\u1fbf"): ord("'"),  # GREEK PSILI
                # ord('\u2019') already included above - RIGHT SINGLE QUOTATION MARK
                ord("\u2032"): ord("'"),  # PRIME
                ord("\ua78c"): ord("'"),  # LATIN SMALL LETTER SALTILLO
                ord("\uff07"): ord("'"),  # FULLWIDTH APOSTROPHE
            }

            # Apply all replacements at once
            processed_text = processed_text.translate(trans_dict)

        # Remove consecutive duplicate lines
        if self.remove_duplicate_lines:
            lines = processed_text.split("\n")
            unique_lines = []
            for i, line in enumerate(lines):
                if i == 0 or line.strip() != lines[i - 1].strip():
                    unique_lines.append(line)
            processed_text = "\n".join(unique_lines)

        # Apply custom rules
        for rule in self.custom_rules:
            pattern = rule.get("pattern")
            replacement = rule.get("replacement")
            flags = rule.get("flags", 0)
            if pattern and replacement is not None:
                processed_text = re.sub(pattern, replacement, processed_text, flags=flags)

        # Restore preserved content
        for i, data in enumerate(special_content):
            placeholder = placeholder_pattern.format(i)
            processed_text = processed_text.replace(placeholder, data)

        return str(processed_text.strip())
