"""
This module contains a collection of utility functions for text processing,
"""

import logging
import re
from typing import List, Dict, Any, Optional

import numpy as np
from haystack.lazy_imports import LazyImport

with LazyImport(
    "Run 'pip install spacy; python -m spacy download en_core_web_sm; python -m spacy download zh_core_web_sm'"
) as spacy_import:
    import spacy

with LazyImport("Run 'pip install tiktoken'") as tiktoken_import:
    import tiktoken

# Global variables to cache spaCy models
_SPACY_MODELS: Dict[str, Any] = {}


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count tokens in text using tiktoken.

    :param text: The input text to count tokens for
    :param model: The model name to use for token counting
    :returns: Number of tokens in the text
    """
    tiktoken_import.check()
    try:
        encoder = tiktoken.encoding_for_model(model)
        return len(encoder.encode(text))
    except (KeyError, ValueError, RuntimeError) as e:
        logging.warning("Token count failed: %s. Using approximation.", e)
        # Fallback to approximation
        return len(text) // 4


def slice_text_by_tokens(
    text: str, start_idx: Optional[int] = None, end_idx: Optional[int] = None, model: str = "gpt-4o"
) -> str:
    """
    Slice text by token indices rather than character indices.

    :param text: The input text to slice
    :param start_idx: The starting token index (inclusive), None means beginning
    :param end_idx: The ending token index (exclusive), None means end
    :param model: The model name to use for tokenization
    :returns: The sliced text corresponding to the token range
    """
    tiktoken_import.check()
    encoder = tiktoken.encoding_for_model(model)
    tokens = encoder.encode(text)
    token_slice = tokens[start_idx:end_idx]
    return encoder.decode(token_slice)


def detect_language(text: str, language: Optional[str] = None) -> str:
    """
    Detect the language of a text.

    :param text: Text to analyze for language detection
    :param language: If provided, returns this language code instead of detecting
    :returns: Language code ('en', 'zh', etc.)
    """
    if language is not None:
        return language
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    if chinese_chars > len(text) * 0.1:  # If >10% is Chinese
        return "zh"
    return "en"


def split_sentences(text: str, language: Optional[str] = None) -> List[str]:
    """
    Split text into sentences using spaCy.

    Uses spaCy's linguistic models for accurate sentence boundary detection
    across languages. Falls back to regex patterns if spaCy fails.

    For optimal results, ensure the appropriate spaCy models are installed::

        python -m spacy download en_core_web_sm
        python -m spacy download zh_core_web_sm

    :param text: Text to split into sentences
    :param language: Language code ('en', 'zh', etc.) or None for auto-detection
    :returns: List of sentences
    """
    spacy_import.check()

    if not text.strip():
        return []

    if language is None:
        language = detect_language(text)

    try:
        # Load and cache spaCy model
        if language not in _SPACY_MODELS:
            _SPACY_MODELS[language] = spacy.load(f"{language}_core_web_sm")

        nlp = _SPACY_MODELS[language]
        doc = nlp(text)
        return [sent.text for sent in doc.sents]
    except (OSError, KeyError, ValueError, RuntimeError) as e:
        # Simple regex fallback if spaCy fails
        print(f"SpaCy error: {e}. Using regex fallback.")
        pattern = r"(?<=[.!?。！？])\s+"
        return [s.strip() for s in re.split(pattern, text) if s.strip()]


def extract_markdown_headings(text: str) -> List[Dict[str, Any]]:
    """
    Extract heading information from Markdown text.

    :param text: Markdown text to extract headings from
    :returns: List of dictionaries with heading information (text, level, position)
    """
    # Create a working copy to modify
    working_text = text
    protected_blocks = {}
    block_counter = 0

    # 1. Protect code blocks
    code_pattern = r"```[\s\S]*?```"
    for match in re.finditer(code_pattern, working_text):
        placeholder = f"PROTECTED_BLOCK_{block_counter}"
        block_counter += 1
        protected_blocks[placeholder] = match.group(0)
        working_text = working_text.replace(match.group(0), placeholder)

    # 2. Protect display math LaTeX blocks (which could contain # symbols)
    display_latex_pattern = r"\$\$[\s\S]*?\$\$"
    for match in re.finditer(display_latex_pattern, working_text):
        placeholder = f"PROTECTED_BLOCK_{block_counter}"
        block_counter += 1
        protected_blocks[placeholder] = match.group(0)
        working_text = working_text.replace(match.group(0), placeholder)

    # 3. Protect LaTeX environments (which could span multiple lines)
    env_pattern = r"\\begin\{[^}]+\}[\s\S]*?\\end\{[^}]+\}"
    for match in re.finditer(env_pattern, working_text):
        placeholder = f"PROTECTED_BLOCK_{block_counter}"
        block_counter += 1
        protected_blocks[placeholder] = match.group(0)
        working_text = working_text.replace(match.group(0), placeholder)

    # Find headings
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    headings = []

    for match in heading_pattern.finditer(working_text):
        level = len(match.group(1))
        heading_text = match.group(2)
        position = match.start()

        # Restore any protected blocks in the heading text
        for placeholder, original in protected_blocks.items():
            heading_text = heading_text.replace(placeholder, original)

        headings.append({"text": heading_text, "level": level, "position": position})

    return headings


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    :param v1: First vector
    :param v2: Second vector
    :returns: Cosine similarity value between -1 and 1
    """

    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0

    return dot_product / (norm_v1 * norm_v2)
