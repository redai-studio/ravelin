import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from haystack.dataclasses import Document

from ravelin.components.preprocessors.markdown_document_splitter import (
    MarkdownDocumentSplitter,
    chunk_by_sentence,
    chunk_by_structure,
    extract_markdown_headings,
    SemanticPreferenceConfig,
)
from ravelin.embedding_functions import RegisteredEmbeddingFunction

TEST_DIR = Path(__file__).parent / "fixtures"
TEST_DOCUMENT_PATH = TEST_DIR / "example_document.md"


class TestMarkdownDocumentSplitter(unittest.TestCase):
    """Tests for the MarkdownDocumentSplitter component."""

    def setUp(self):
        """Set up test fixtures."""
        # Load the test document
        with open(TEST_DOCUMENT_PATH, "r", encoding="utf-8") as f:
            self.test_content = f.read()

        # Create a mock embedding function for semantic splitting
        self.mock_embedding_function = MagicMock(spec=RegisteredEmbeddingFunction)

        # Create instance for testing
        self.splitter = MarkdownDocumentSplitter(
            strategy="semantic",
            max_tokens=3000,
            min_tokens=1500,
            token_encoding_model="gpt-3.5-turbo",
            overlap_tokens=50,
            embedding_function=self.mock_embedding_function,
            semantic_preferences={"heading_bonus": 1.5},
        )

    def test_to_dict(self):
        # Test the to_dict method
        splitter_dict = self.splitter.to_dict()

        # Assert the overall structure
        self.assertIn("init_parameters", splitter_dict)
        self.assertIn("type", splitter_dict)

        # Check init parameters
        init_params = splitter_dict["init_parameters"]
        self.assertEqual("semantic", init_params["strategy"])
        self.assertEqual(3000, init_params["max_tokens"])
        self.assertEqual(1500, init_params["min_tokens"])
        self.assertEqual("gpt-3.5-turbo", init_params["token_encoding_model"])
        self.assertEqual(50, init_params["overlap_tokens"])
        self.assertEqual(self.mock_embedding_function.to_dict.return_value, init_params["embedding_function"])

        # Check semantic preferences were properly serialized
        self.assertIn("semantic_preferences", init_params)
        self.assertEqual(1.5, init_params["semantic_preferences"]["heading_bonus"])

    @patch("ravelin.embedding_functions.EmbeddingFunctionFactory.create_from_dict")
    def test_from_dict_happy_path(self, mock_create_from_dict):
        # Set up the mock to return our embedding function
        mock_create_from_dict.return_value = self.mock_embedding_function

        # Create test data
        data = {
            "type": "ravelin.components.preprocessors.markdown_document_splitter.MarkdownDocumentSplitter",
            "init_parameters": {
                "strategy": "semantic",
                "max_tokens": 3000,
                "min_tokens": 1500,
                "token_encoding_model": "gpt-3.5-turbo",
                "overlap_tokens": 50,
                "embedding_function": {"name": "test-embedding-function", "type": "model-service"},
                "semantic_preferences": {"heading_bonus": 0.9, "dynamic_threshold_base": 0.6},
            },
        }

        # Call from_dict
        splitter = MarkdownDocumentSplitter.from_dict(data)

        # Assertions
        self.assertIsInstance(splitter, MarkdownDocumentSplitter)
        self.assertEqual("semantic", splitter.strategy)
        self.assertEqual(3000, splitter.max_tokens)
        self.assertEqual(1500, splitter.min_tokens)
        self.assertEqual("gpt-3.5-turbo", splitter.token_encoding_model)
        self.assertEqual(50, splitter.overlap_tokens)
        self.assertEqual(self.mock_embedding_function, splitter.embedding_function)
        self.assertEqual(0.9, splitter.semantic_preferences.heading_bonus)
        self.assertEqual(0.6, splitter.semantic_preferences.dynamic_threshold_base)

        # Verify EmbeddingFunctionFactory.create_from_dict was called
        mock_create_from_dict.assert_called_once_with({"name": "test-embedding-function", "type": "model-service"})

    @patch("ravelin.embedding_functions.EmbeddingFunctionFactory.create_from_dict")
    def test_from_dict_with_invalid_embedding_function_type(self, mock_create_from_dict):
        # Return an invalid function type
        mock_create_from_dict.side_effect = ValueError("Unknown embedding function type: 'unknown'")

        data = {
            "type": "ravelin.components.markdown_document_splitter.MarkdownDocumentSplitter",
            "init_parameters": {
                "strategy": "semantic",
                "max_tokens": 4000,
                "min_tokens": 2000,
                "token_encoding_model": "gpt-4o",
                "embedding_function": {"name": "invalid-function-type", "type": "unknown"},
            },
        }

        # Test that it raises a ValueError for missing embedding_function
        with self.assertRaises(ValueError):
            MarkdownDocumentSplitter.from_dict(data)

    def test_sentence_chunking_real_document(self):
        """Test sentence-based chunking with a real document."""
        # Create the splitter with sentence strategy
        splitter = MarkdownDocumentSplitter(
            strategy="sentence",
            max_tokens=100,
            min_tokens=50,
            overlap_tokens=10,  # Small enough to ensure multiple chunks
        )

        # Process the document
        doc = Document(content=self.test_content)
        result = splitter.run([doc])
        chunks = result["documents"]

        # Check that we got multiple chunks
        self.assertGreater(len(chunks), 1, "Document should be split into multiple chunks")

        # Check that chunks have content
        for chunk in chunks:
            self.assertGreater(len(chunk.content), 0, "Chunks should have content")

        # Check for overlap between chunks
        if len(chunks) > 1:
            # Get last few words from first chunk
            first_chunk_words = chunks[0].content.split()[-5:]
            # Get first few words from second chunk
            second_chunk_words = chunks[1].content.split()[:10]

            # Check if there's some overlap
            overlap_found = any(word in second_chunk_words for word in first_chunk_words)
            self.assertTrue(overlap_found, "Should have some overlap between chunks")

    def test_structural_chunking_real_document(self):
        """Test structural chunking with a real document."""
        # Create the splitter with structural strategy
        splitter = MarkdownDocumentSplitter(
            strategy="structural", max_tokens=200, min_tokens=100  # Small enough to ensure multiple chunks
        )

        # Process the document
        doc = Document(content=self.test_content)
        result = splitter.run([doc])
        chunks = result["documents"]

        # Check that we got multiple chunks
        self.assertGreater(len(chunks), 1, "Document should be split into multiple chunks")

        # Check that headings are preserved at the start of chunks
        for chunk in chunks:
            if "# " in chunk.content or "## " in chunk.content or "### " in chunk.content:
                # If chunk contains a heading, it should be at the beginning or have a heading structure
                first_line = chunk.content.strip().split("\n")[0].strip()
                self.assertTrue(
                    first_line.startswith("#"), f"Chunks should preserve heading structure. First line: '{first_line}'"
                )

        # Check metadata structure
        for chunk in chunks:
            self.assertIn("heading", chunk.meta, "Structural chunks should have heading metadata")
            self.assertIn("level", chunk.meta, "Structural chunks should have level metadata")

    @patch("ravelin.components.preprocessors.markdown_document_splitter.count_tokens")
    def test_semantic_chunking_real_document(self, mock_count_tokens):
        """Test semantic chunking with a real document."""
        # Make count_tokens return realistic values
        mock_count_tokens.side_effect = lambda text, model: len(text.split()) * 1.3

        # Prepare a real embeddings response
        def mock_embed(texts):
            embeddings = []
            for text in texts:
                # Create distinct embeddings for different sections
                if "Technical" in text or "technical" in text:
                    embeddings.append([0.8, 0.1, 0.1])  # Technical content direction
                elif "Business" in text or "business" in text:
                    embeddings.append([0.1, 0.8, 0.1])  # Business content direction
                elif "Conclusion" in text:
                    embeddings.append([0.1, 0.1, 0.8])  # Conclusion content direction
                else:
                    embeddings.append([0.33, 0.33, 0.33])  # General content
            return embeddings

        # Configure the mock embedding function
        self.mock_embedding_function.side_effect = mock_embed

        # Create the splitter with semantic strategy and aggressive preferences
        splitter = MarkdownDocumentSplitter(
            strategy="semantic",
            max_tokens=200,
            min_tokens=100,
            embedding_function=self.mock_embedding_function,
            semantic_preferences=None,
        )

        # Process the document
        doc = Document(content=self.test_content)
        result = splitter.run([doc])
        chunks = result["documents"]

        # for chunk in chunks:
        #     print(f"Chunk content: {chunk.content}")
        #     print(f"Chunk meta: {chunk.meta}")
        #     print("-" * 50)

        # Check that we got multiple chunks
        self.assertGreater(len(chunks), 1, "Document should be split into multiple chunks")

        # Check the metadata structure
        for chunk in chunks:
            self.assertIn("semantic", chunk.meta, "Semantic chunks should have semantic flag")
            self.assertTrue(chunk.meta["semantic"], "Semantic flag should be True")

        # Check for semantic coherence - technical and business content should be separate
        technical_chunks = []
        business_chunks = []

        for chunk in chunks:
            if "Technical" in chunk.content:
                technical_chunks.append(chunk)
            if "Business" in chunk.content:
                business_chunks.append(chunk)

        if technical_chunks and business_chunks:
            # Check that technical and business sections aren't in the same chunk
            for tech_chunk in technical_chunks:
                for bus_chunk in business_chunks:
                    self.assertFalse(
                        tech_chunk is bus_chunk, "Technical and Business sections should not be in the same chunk"
                    )


class TestMarkdownDocumentSplitterParameterized(unittest.TestCase):
    """Parameterized tests for better coverage of MarkdownDocumentSplitter."""

    def test_initialization_parameters(self):
        """Test initialization with various parameter combinations."""
        mock_embedding_function = MagicMock(spec=RegisteredEmbeddingFunction)

        test_cases = [
            # strategy, max_tokens, min_tokens, token_model, overlap, embedding_function, semantic_prefs, expected_valid
            ("sentence", 1000, 500, "gpt-4", 0, None, None, True),  # Basic sentence strategy
            ("sentence", 1000, 500, "gpt-4", 100, None, None, True),  # Sentence with overlap
            ("structural", 2000, 1000, "gpt-4", 0, None, None, True),  # Basic structural
            ("semantic", 3000, 1500, "gpt-4", 0, mock_embedding_function, None, True),  # With embedding function
            (
                "semantic",
                3000,
                1500,
                "gpt-4",
                0,
                mock_embedding_function,
                {"heading_bonus": 0.9},
                True,
            ),  # With dict prefs
            (
                "semantic",
                3000,
                1500,
                "gpt-4",
                0,
                mock_embedding_function,
                SemanticPreferenceConfig(heading_bonus=0.9),
                True,
            ),  # With config object
            ("semantic", 3000, 1500, "gpt-4", 0, None, None, False),  # Invalid: semantic without embedding
            ("invalid", 1000, 500, "gpt-4", 0, None, None, False),  # Invalid strategy
        ]

        for idx, (
            strategy,
            max_tokens,
            min_tokens,
            model,
            overlap_tokens,
            embedding_function,
            prefs,
            expected_valid,
        ) in enumerate(test_cases):
            with self.subTest(case=idx, strategy=strategy, max_tokens=max_tokens):
                if expected_valid:
                    try:
                        splitter = MarkdownDocumentSplitter(
                            strategy=strategy,
                            max_tokens=max_tokens,
                            min_tokens=min_tokens,
                            token_encoding_model=model,
                            overlap_tokens=overlap_tokens,
                            embedding_function=embedding_function,
                            semantic_preferences=prefs,
                        )
                        self.assertEqual(strategy, splitter.strategy)
                        self.assertEqual(max_tokens, splitter.max_tokens)
                        self.assertEqual(min_tokens, splitter.min_tokens)
                        self.assertEqual(model, splitter.token_encoding_model)
                        self.assertEqual(overlap_tokens, splitter.overlap_tokens)
                        self.assertEqual(embedding_function, splitter.embedding_function)
                    except ValueError:
                        self.fail(f"Valid configuration raised ValueError: {strategy}")
                else:
                    with self.assertRaises(ValueError):
                        MarkdownDocumentSplitter(
                            strategy=strategy,
                            max_tokens=max_tokens,
                            min_tokens=min_tokens,
                            token_encoding_model=model,
                            overlap_tokens=overlap_tokens,
                            embedding_function=embedding_function,
                            semantic_preferences=prefs,
                        )

    @patch("ravelin.components.preprocessors.markdown_document_splitter.count_tokens")
    def test_sentence_chunking_parameterized(self, mock_count_tokens):
        """Test sentence chunking with various parameters and edge cases."""
        # Set up a realistic token counting function
        mock_count_tokens.side_effect = lambda text, model: len(text.split())

        test_cases = [
            # test_name, max_tokens, overlap, content, expected_chunks
            (
                "basic_split",
                5,  # Small max_tokens to force splitting
                0,  # No overlap
                "Sentence one. Sentence two. Sentence three. Sentence four.",
                2,  # Expect 2 chunks with these small token limits
            ),
            (
                "with_overlap",
                5,
                2,  # Significant overlap
                "Sentence one. Sentence two here. Sentence three. Sentence four.",
                2,  # 2 chunks but with overlap
            ),
            (
                "single_chunk",
                100,  # Large max_tokens
                0,
                "Short document with a few sentences.",
                1,  # Should fit in one chunk
            ),
            (
                "paragraphs",
                8,  # Force paragraph breaks
                0,
                "Para one has\nsome text.\n\nPara two has\nmore text.\n\nPara three.",
                2,  # Should break into 2 chunks
            ),
            ("empty_text", 100, 0, "", 0),  # Should return empty list
            (
                "very_large_sentence",
                5,  # Too small for the sentence
                0,
                "This is an extremely long sentence that exceeds the max token limit and cannot be broken further.",
                1,  # Should still create one chunk despite exceeding max
            ),
            ("edge_case_single_word", 10, 0, "Word.", 1),  # Single word should be one chunk
        ]

        for test_name, max_tokens, overlap_tokens, content, expected_chunk_count in test_cases:
            with self.subTest(name=test_name):
                chunks = chunk_by_sentence(content, max_tokens, overlap_tokens, "gpt-4")
                if test_name == "with_overlap":
                    print(chunks)

                self.assertEqual(
                    expected_chunk_count,
                    len(chunks),
                    f"Expected {expected_chunk_count} chunks for test '{test_name}', got {len(chunks)}",
                )

                # For overlap tests, check overlap content when applicable
                if overlap_tokens > 0 and len(chunks) > 1:
                    # Take the first two chunks to check overlap
                    first_chunk_content = chunks[0]["content"]
                    second_chunk_content = chunks[1]["content"]

                    # Check for some content from first chunk in the beginning of second chunk
                    has_overlap = any(
                        word in second_chunk_content.split()[:10]
                        for word in first_chunk_content.split()[-5:]
                        if len(word) > 3  # Ignore small words that might appear naturally
                    )

                    self.assertTrue(has_overlap, f"Expected overlap not found between chunks for test '{test_name}'")

    @patch("ravelin.components.preprocessors.markdown_document_splitter.count_tokens")
    @patch("ravelin.components.preprocessors.markdown_document_splitter.extract_markdown_headings")
    def test_structural_chunking_parameterized(self, mock_headings, mock_count_tokens):
        """Test structural chunking with various parameters and edge cases."""
        # Test cases where we don't override extract_markdown_headings
        test_cases_normal = [
            # test_name, max_tokens, min_tokens, content, expected_chunks
            (
                "basic_headings",
                10,  # Small enough to force splits
                5,  # Reasonable minimum
                "# Heading 1\nContent 1\n\n# Heading 2\nContent 2",
                2,  # Expect 2 chunks, one per heading
            ),
            (
                "no_headings",
                6,
                3,
                "Paragraph one with content.\n\nParagraph two with content.",
                2,  # Should use sentence chunking fallback
            ),
            ("empty_document", 10, 5, "", 0),  # Empty document
            ("single_heading_tiny", 100, 10, "# Just a heading", 1),  # Large limit  # Small minimum  # Single heading
        ]

        # For normal cases, let the extract_markdown_headings function work naturally
        mock_headings.side_effect = lambda text: extract_markdown_headings_original(text)

        for test_name, max_tokens, min_tokens, content, expected_chunk_count in test_cases_normal:
            with self.subTest(name=test_name):
                # Setup token counting to be deterministic
                mock_count_tokens.side_effect = lambda text, model: len(text.split())

                chunks = chunk_by_structure(content, max_tokens, min_tokens, "gpt-4")
                self.assertEqual(
                    expected_chunk_count, len(chunks), f"Expected {expected_chunk_count} chunks for test '{test_name}'"
                )

                # Check that metadata has expected fields for non-empty results
                if len(chunks) > 0:
                    if "#" in content:  # If there are headings
                        self.assertIn("heading", chunks[0]["metadata"], "Chunk should have heading metadata")
                        self.assertIn("level", chunks[0]["metadata"], "Chunk should have level metadata")

        # Test cases where we control heading extraction precisely
        test_cases_custom = [
            # test_name, max_tokens, min_tokens, content, mock_headings_return, expected_chunks
            (
                "nested_headings",
                8,
                4,
                "# Main\nIntro\n\n## Sub1\nContent1\n\n## Sub2\nContent2",
                [
                    {"level": 1, "text": "Main", "position": 0},  # 3 tokens
                    {"level": 2, "text": "Sub1", "position": 12},  # 3 tokens
                    {"level": 2, "text": "Sub2", "position": 30},  # 3 tokens
                ],
                2,  # Should create 2 chunks due to headings and min_token restrictions
            ),
            (
                "huge_section",
                6,  # Very small to force splits
                3,
                "# H1\nThis is a very long section. It exceeds the token limit.",
                [
                    {"level": 1, "text": "H1", "position": 0},
                ],
                2,  # Should split the large section
            ),
            (
                "small_sections",
                20,  # Larger limit
                10,  # Higher min to force merging small sections
                "# H1\nTiny.\n\n# H2\nAlso tiny.\n\n# H3\nTiny too.",
                [
                    {"level": 1, "text": "H1", "position": 0},
                    {"level": 1, "text": "H2", "position": 10},
                    {"level": 1, "text": "H3", "position": 25},
                ],
                1,  # Should merge small sections
            ),
        ]

        for test_name, max_tokens, min_tokens, content, headings_return, expected_chunk_count in test_cases_custom:
            with self.subTest(name=f"custom_{test_name}"):
                # Override with custom heading extraction
                mock_headings.return_value = headings_return

                # Setup token counting to be deterministic
                mock_count_tokens.side_effect = lambda text, model: len(text.split())

                chunks = chunk_by_structure(content, max_tokens, min_tokens, "gpt-4")
                self.assertEqual(
                    expected_chunk_count, len(chunks), f"Expected {expected_chunk_count} chunks for test '{test_name}'"
                )

    def test_semantic_preferences_parameterized(self):
        """Test creating SemanticPreferenceConfig with various parameters."""
        test_cases = [
            # test_name, params_dict, expected_values
            (
                "default_values",
                {},  # Empty dict = use defaults
                {
                    "heading_bonus": 0.4,
                    "strong_break_threshold": 0.5,
                    "exceptional_break_threshold": 0.8,
                    "dynamic_threshold_base": 0.5,
                    "min_chunk_ratio": 0.7,
                },
            ),
            (
                "custom_values",
                {
                    "heading_bonus": 0.7,
                    "strong_break_threshold": 0.6,
                    "exceptional_break_threshold": 0.9,
                    "dynamic_threshold_base": 0.4,
                    "min_chunk_ratio": 0.6,
                },
                {
                    "heading_bonus": 0.7,
                    "strong_break_threshold": 0.6,
                    "exceptional_break_threshold": 0.9,
                    "dynamic_threshold_base": 0.4,
                    "min_chunk_ratio": 0.6,
                },
            ),
            (
                "partial_override",
                {"heading_bonus": 0.9, "min_chunk_ratio": 0.5},
                {
                    "heading_bonus": 0.9,
                    "strong_break_threshold": 0.5,  # Default
                    "exceptional_break_threshold": 0.8,  # Default
                    "dynamic_threshold_base": 0.5,  # Default
                    "min_chunk_ratio": 0.5,
                },
            ),
        ]

        for test_name, params, expected in test_cases:
            with self.subTest(name=test_name):
                config = SemanticPreferenceConfig(**params)
                for key, value in expected.items():
                    self.assertEqual(value, getattr(config, key), f"Expected {key}={value} for test '{test_name}'")

    @patch("ravelin.components.preprocessors.markdown_document_splitter.count_tokens")
    def test_run_with_empty_documents(self, mock_count_tokens):
        """Test handling of empty documents in the run method."""
        mock_count_tokens.return_value = 0

        test_cases = [
            # strategy, docs, expected_chunk_count
            ("sentence", [], 0),  # Empty list
            ("sentence", [Document(content="")], 0),  # Single empty document
            ("structural", [], 0),  # Empty list
            ("structural", [Document(content="")], 0),  # Single empty document
        ]

        for strategy, docs, expected_count in test_cases:
            with self.subTest(strategy=strategy):
                splitter = MarkdownDocumentSplitter(strategy=strategy)
                result = splitter.run(docs)
                self.assertEqual(
                    expected_count,
                    len(result["documents"]),
                    f"Expected {expected_count} chunks for empty document with {strategy} strategy",
                )


# Helper function to avoid mocking complexity
def extract_markdown_headings_original(text):
    """Original implementation for test use"""
    return extract_markdown_headings(text)


if __name__ == "__main__":
    unittest.main()
