import unittest
from haystack.dataclasses import Document

from ravelin.components.preprocessors import MarkdownDocumentCleaner


class TestMarkdownDocumentCleaner(unittest.TestCase):
    def setUp(self):
        self.cleaner = MarkdownDocumentCleaner()

    def test_basic_cleaning(self):
        docs = [Document(content="#Title\n\n\n- Item1")]
        expected = "# Title\n\n- Item1"
        result = self.cleaner.run(documents=docs)
        self.assertEqual(expected, result["documents"][0].content)

    def test_fix_headings(self):
        docs = [Document(content="#Title\n##Subtitle\n###Third level")]
        expected = "# Title\n## Subtitle\n### Third level"
        result = self.cleaner.run(documents=docs)
        self.assertEqual(expected, result["documents"][0].content)

    def test_fix_lists(self):
        docs = [Document(content="- Item1\n*Item2\n1.Item3\n- [x]Task")]
        expected = "- Item1\n* Item2\n1. Item3\n- [x] Task"
        result = self.cleaner.run(documents=docs)
        self.assertEqual(expected, result["documents"][0].content)

    def test_normalize_quotes(self):
        # Test with Unicode quotes from different languages
        docs = [Document(content="\u201cDouble quotes\u201d and \u02bbsingle quotes\u2019")]
        expected = "\"Double quotes\" and 'single quotes'"
        result = self.cleaner.run(documents=docs)
        self.assertEqual(expected, result["documents"][0].content)

    def test_remove_html_comments(self):
        docs = [Document(content="Before <!-- comment --> After")]
        expected = "Before  After"
        result = self.cleaner.run(documents=docs)
        self.assertEqual(expected, result["documents"][0].content)

    def test_remove_duplicate_lines(self):
        docs = [Document(content="Line 1\nLine 1\nLine 2\nLine 2")]
        expected = "Line 1\nLine 2"
        result = self.cleaner.run(documents=docs)
        self.assertEqual(expected, result["documents"][0].content)

    def test_preserve_fenced_code_blocks(self):
        content = "Text\n```python\nprint('Hello')\nprint(\"quotes shouldn\u2019t change\")\n```\nMore text"
        docs = [Document(content=content)]
        result = self.cleaner.run(documents=docs)
        self.assertEqual(content, result["documents"][0].content)

    def test_preserve_inline_code(self):
        content = "Text `code with \u201cquotes\u201d inside` more text"
        docs = [Document(content=content)]
        result = self.cleaner.run(documents=docs)
        # Check if code block with quotes is unchanged
        self.assertIn("`code with \u201cquotes\u201d inside`", result["documents"][0].content)

    def test_preserve_indented_code_blocks(self):
        content = "Normal text\n\n    # indented code\n    print('Hello')\n    print('Hello')\n\nMore text"
        docs = [Document(content=content)]
        result = self.cleaner.run(documents=docs)
        self.assertEqual(content, result["documents"][0].content)

    def test_preserve_latex_block(self):
        content = "Formula: $$E = mc^2 + \u201cquoted\u201d$$"
        docs = [Document(content=content)]
        result = self.cleaner.run(documents=docs)
        self.assertEqual(content, result["documents"][0].content)

    def test_preserve_latex_inline(self):
        content = "Formula: $E = \u201cmc^2\u201d$ is famous"
        docs = [Document(content=content)]
        result = self.cleaner.run(documents=docs)
        self.assertEqual(content, result["documents"][0].content)

    def test_custom_rules(self):
        custom_rule = {"pattern": r"TODO", "replacement": "DONE", "flags": 0}
        cleaner = MarkdownDocumentCleaner(custom_rules=[custom_rule])
        docs = [Document(content="Remember to TODO this item")]
        expected = "Remember to DONE this item"
        result = cleaner.run(documents=docs)
        self.assertEqual(expected, result["documents"][0].content)

    def test_null_input(self):
        docs = [Document(content=None)]
        expected = ""
        result = self.cleaner.run(documents=docs)
        self.assertEqual(expected, result["documents"][0].content)

    def test_feature_toggling(self):
        cleaner = MarkdownDocumentCleaner(fix_headings=False, remove_html_comments=False)
        docs = [Document(content="#Title <!-- comment -->")]
        expected = "#Title <!-- comment -->"
        result = cleaner.run(documents=docs)
        self.assertEqual(expected, result["documents"][0].content)

    def test_complex_mixed_content(self):
        # This test verifies that the cleaner correctly handles a mix of content types
        content = """#Document with mixed elements

Normal text with “smart quotes❞.

```python
# Code with "quotes" - should be preserved
if x == 'string':
    print('Hello')
    print('Hello')  # Duplicate line in code should be preserved
```
Formula: $E = \u201cmc^2\u201d$ should preserve quotes.

<!-- This comment should be removed -->

- List item1
-Item2

    # Indented code - should be preserved
    x = 1
    x = 1
    # Duplicate line in indented code
"""
        docs = [Document(content=content)]
        result = self.cleaner.run(documents=docs)

        # Check specific expected outcomes
        self.assertIn("# Document with mixed elements", result["documents"][0].content)
        self.assertIn('Normal text with "smart quotes"', result["documents"][0].content)
        self.assertIn('```python\n# Code with "quotes"', result["documents"][0].content)
        self.assertIn("print('Hello')\n    print('Hello')", result["documents"][0].content)
        self.assertIn("$E = \u201cmc^2\u201d$", result["documents"][0].content)
        self.assertNotIn("<!-- This comment should be removed -->", result["documents"][0].content)
        self.assertIn("- List item1\n- Item2", result["documents"][0].content)
        self.assertIn("    # Indented code - should be preserved", result["documents"][0].content)
        self.assertIn("    x = 1\n    x = 1", result["documents"][0].content)


if __name__ == "__main__":
    unittest.main()
