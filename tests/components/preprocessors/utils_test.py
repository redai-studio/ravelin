import unittest
from unittest.mock import patch, Mock
from ravelin.components.preprocessors.utils import (
    count_tokens,
    slice_text_by_tokens,
    split_sentences,
    detect_language,
    extract_markdown_headings,
    cosine_similarity,
)
import numpy as np


class TestCountTokens(unittest.TestCase):
    @patch("tiktoken.encoding_for_model")
    def test_count_tokens_with_valid_model(self, mock_encoding_for_model):
        # Setup
        mock_encoder = Mock()
        mock_encoder.encode.return_value = [1, 2, 3, 4]  # 4 tokens
        mock_encoding_for_model.return_value = mock_encoder

        # Test
        result = count_tokens("This is a test", "gpt-4o")

        # Assert
        self.assertEqual(4, result)
        mock_encoding_for_model.assert_called_once_with("gpt-4o")
        mock_encoder.encode.assert_called_once_with("This is a test")

    @patch("tiktoken.encoding_for_model")
    @patch("logging.warning")
    def test_count_tokens_with_error(self, mock_warning, mock_encoding_for_model):
        mock_encoding_for_model.side_effect = ValueError("Test error")

        # Test
        result = count_tokens("This is a test", "invalid-model")

        # Assert
        self.assertEqual(3, result)  # len("This is a test") // 4 = 3
        mock_warning.assert_called_once()


class TestSliceTextByTokens(unittest.TestCase):
    @patch("tiktoken.encoding_for_model")
    def test_slice_text_by_tokens(self, mock_encoding_for_model):
        # Setup
        mock_encoder = Mock()
        mock_encoder.encode.side_effect = lambda text: text.split()
        mock_encoder.decode.side_effect = lambda tokens: " ".join(tokens)
        mock_encoding_for_model.return_value = mock_encoder

        text = "This is a test sentence with multiple words"
        result = slice_text_by_tokens(text, 2, 4, "gpt-4o")
        self.assertEqual("a test", result)


class TestDetectLanguage(unittest.TestCase):
    def test_detect_language_with_explicit_language(self):
        # Test
        result = detect_language("This is English text", language="fr")

        # Assert
        self.assertEqual("fr", result)

    def test_detect_language_english(self):
        # Test
        result = detect_language("This is English text")

        # Assert
        self.assertEqual("en", result)

    def test_detect_language_chinese(self):
        # Test
        result = detect_language("这是中文文本")

        # Assert
        self.assertEqual("zh", result)

    def test_detect_language_mixed_with_chinese_threshold(self):
        # Test with text that has >10% Chinese characters
        result = detect_language("Text with 这些中文字符 in it")

        # Assert
        self.assertEqual("zh", result)


class TestSplitSentences(unittest.TestCase):
    @patch("ravelin.components.preprocessors.utils._SPACY_MODELS", {})
    def test_split_sentences_empty_text(self):
        result = split_sentences("")

        # Assert
        self.assertEqual([], result)

    @patch("ravelin.components.preprocessors.utils._SPACY_MODELS", {})
    @patch("spacy.load")
    def test_split_sentences_english(self, mock_load):
        # Setup
        mock_doc = Mock()
        mock_sent1 = Mock()
        mock_sent1.text = "This is sentence one."
        mock_sent2 = Mock()
        mock_sent2.text = "This is sentence two."
        mock_doc.sents = [mock_sent1, mock_sent2]

        mock_nlp = Mock()
        mock_nlp.return_value = mock_doc
        mock_load.return_value = mock_nlp

        # Test
        result = split_sentences("This is sentence one. This is sentence two.", language="en")

        # Assert
        self.assertEqual(["This is sentence one.", "This is sentence two."], result)
        mock_load.assert_called_once_with("en_core_web_sm")

    @patch("ravelin.components.preprocessors.utils._SPACY_MODELS", {})
    @patch("spacy.load")
    def test_split_sentences_fallback_regex(self, mock_load):
        # Setup
        mock_load.side_effect = ValueError("Test error")

        # Test
        result = split_sentences("Sentence one. Sentence two!", language="en")

        # Assert
        self.assertEqual(["Sentence one.", "Sentence two!"], result)

    @patch("ravelin.components.preprocessors.utils._SPACY_MODELS", {})
    @patch("ravelin.components.preprocessors.utils.detect_language")
    @patch("spacy.load")
    def test_split_sentences_auto_language_detection(self, mock_load, mock_detect_language):
        # Setup
        mock_detect_language.return_value = "en"
        mock_doc = Mock()
        mock_sent = Mock()
        mock_sent.text = "This is a test."
        mock_doc.sents = [mock_sent]

        mock_nlp = Mock()
        mock_nlp.return_value = mock_doc
        mock_load.return_value = mock_nlp

        # Test
        result = split_sentences("This is a test.")

        # Assert
        self.assertEqual(["This is a test."], result)
        mock_detect_language.assert_called_once_with("This is a test.")
        mock_load.assert_called_once_with("en_core_web_sm")


class TestExtractMarkdownHeadings(unittest.TestCase):
    def test_extract_simple_headings(self):
        # Test
        text = """
# Heading 1
Some content
## Heading 2
More content
### Heading 3
"""
        result = extract_markdown_headings(text)

        # Assert
        self.assertEqual(3, len(result))
        self.assertEqual("Heading 1", result[0]["text"])
        self.assertEqual(1, result[0]["level"])
        self.assertEqual("Heading 2", result[1]["text"])
        self.assertEqual(2, result[1]["level"])
        self.assertEqual("Heading 3", result[2]["text"])
        self.assertEqual(3, result[2]["level"])

    def test_extract_headings_with_code_blocks(self):
        # Test
        text = """
# Heading 1
```
# This is code, not a heading
```
## Heading 2
"""
        result = extract_markdown_headings(text)

        # Assert
        self.assertEqual(2, len(result))
        self.assertEqual("Heading 1", result[0]["text"])
        self.assertEqual("Heading 2", result[1]["text"])

    def test_extract_headings_with_latex(self):
        # Test
        text = """
# Heading with $$ y = x^2 $$ math
## Normal heading
$$
# This is in display math, not a heading
$$
"""
        result = extract_markdown_headings(text)

        # Assert
        self.assertEqual(2, len(result))
        self.assertEqual("Heading with $$ y = x^2 $$ math", result[0]["text"])
        self.assertEqual("Normal heading", result[1]["text"])


class TestCosineSimilarity(unittest.TestCase):
    def test_cosine_similarity_perpendicular_vectors(self):
        # Test
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        result = cosine_similarity(v1, v2)

        # Assert
        self.assertEqual(0.0, result)

    def test_cosine_similarity_identical_vectors(self):
        # Test
        v1 = np.array([1.0, 2.0, 3.0])
        result = cosine_similarity(v1, v1)

        # Assert
        self.assertAlmostEqual(1.0, result)

    def test_cosine_similarity_opposite_vectors(self):
        # Test
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([-1.0, -2.0, -3.0])
        result = cosine_similarity(v1, v2)

        # Assert
        self.assertAlmostEqual(-1.0, result)

    def test_cosine_similarity_zero_vector(self):
        # Test
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([1.0, 2.0, 3.0])
        result = cosine_similarity(v1, v2)

        # Assert
        self.assertEqual(0.0, result)


if __name__ == "__main__":
    unittest.main()
