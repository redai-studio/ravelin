"""
This module contains the `MarkdownDocumentSplitter` class which is used to split Markdown documents into smaller chunks
via various strategies, as well as a bunch of standalone functions for different chunking strategies.
"""

import dataclasses
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Literal, Union

import numpy as np
from haystack import component, default_to_dict, default_from_dict
from haystack.dataclasses import Document
from typing_extensions import Self

from ravelin.components.preprocessors.utils import (
    count_tokens,
    slice_text_by_tokens,
    split_sentences,
    extract_markdown_headings,
    cosine_similarity,
)
from ravelin.embedding_functions import EmbeddingFunction, RegisteredEmbeddingFunction, EmbeddingFunctionFactory

logger = logging.getLogger(__name__)


@dataclass
class SemanticPreferenceConfig:
    """
    Configuration for semantic preferences.

    :param heading_bonus: Extra score for breaking at headings (default: 0.4)
        Higher values create more breaks at headings
    :param strong_break_threshold: Minimum dissimilarity to consider a break (default: 0.5)
        Higher values make the algorithm more conservative with breaks
    :param exceptional_break_threshold: Dissimilarity that forces a break (default: 0.8)
        Higher values require more dramatic topic shifts to break immediately
    :param dynamic_threshold_base: Base threshold for dynamic sizing (default: 0.5)
        Lower values favor larger chunks when content is coherent
    :param min_chunk_ratio: Minimum ratio of min_tokens for a standalone chunk (default: 0.7)
        Higher values are more aggressive about merging small chunks
    """

    heading_bonus: float = 0.4
    strong_break_threshold: float = 0.5
    exceptional_break_threshold: float = 0.8
    dynamic_threshold_base: float = 0.5
    min_chunk_ratio: float = 0.7


@component
class MarkdownDocumentSplitter:
    """
    A preprocessing component that splits documents into smaller chunks according to different strategies:

    - Sentence-based splitting
    - Markdown structure-based splitting
    - Semantic splitting based on embedding similarity and coherence

    This component is often used to breakdown large documents into smaller, more manageable pieces for further
    processing by downstream components, such as semantic search or directly feeding into a language model.

    The component can be configured to have a maximum and minimum number of tokens per chunk, although
    in practice, this boundary might not be strictly enforced for certain splitting strategies.

    :param strategy: The splitting strategy to use. Options are:
        - "sentence": Splits based on sentence boundaries.
        - "structural": Splits based on markdown structure (headers, paragraphs).
        - "semantic": Splits based on semantic similarity using embeddings.
    :param max_tokens: The maximum number of tokens per chunk. Default is 4000.
    :param min_tokens: The minimum number of tokens per chunk. Default is 2000.
    :param token_encoding_model: The model used for token encoding. Default is "gpt-4o".
    :param overlap_tokens: The number of overlapping tokens between chunks.
        Only used in sentence splitting. Default is 0.
    :param embedding_function: The embedding function to use for semantic splitting. This is used to convert text into
        embeddings, which are then used to determine semantic boundaries based on similarity and coherence.
    :param semantic_preferences: Only used in semantic splitting. Advanced configuration for semantic preferences, such
        as heading bonuses and thresholds. Check `SemanticPreferenceConfig` for more details.
    """

    def __init__(
        self,
        strategy: Literal["sentence", "structural", "semantic"] = "semantic",
        max_tokens: int = 4000,
        min_tokens: int = 2000,
        token_encoding_model: str = "gpt-4o",
        *,
        overlap_tokens: int = 0,
        embedding_function: Optional[EmbeddingFunction] = None,
        semantic_preferences: Union[Dict[str, Any], SemanticPreferenceConfig, None] = None,
    ):
        self.strategy = strategy
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.token_encoding_model = token_encoding_model
        self.overlap_tokens = overlap_tokens
        self.embedding_function = embedding_function
        if isinstance(semantic_preferences, dict):
            self.semantic_preferences = SemanticPreferenceConfig(**semantic_preferences)
        else:
            self.semantic_preferences = semantic_preferences or SemanticPreferenceConfig()
        self._validate_strategy()

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a serialized dictionary representation of the component.

        :return: A dictionary representation of the component.
        """
        if not isinstance(self.embedding_function, RegisteredEmbeddingFunction):
            raise ValueError("Embedding function must be a registered embedding function to be serialized.")
        return default_to_dict(
            self,
            strategy=self.strategy,
            max_tokens=self.max_tokens,
            min_tokens=self.min_tokens,
            token_encoding_model=self.token_encoding_model,
            overlap_tokens=self.overlap_tokens,
            embedding_function=self.embedding_function,
            semantic_preferences=dataclasses.asdict(self.semantic_preferences),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Loads a MarkdownDocumentSplitter component from its serialized dictionary representation.

        :param data: The serialized dictionary representation of the component.
        :return: The deserialized MarkdownDocumentSplitter component.
        """
        # Make a shallow copy of init_parameters to avoid modifying the original data
        init_parameters_data = data.get("init_parameters", {}).copy()
        if init_parameters_data.get("embedding_function"):
            # Load the embedding function from the factory
            embedding_function = EmbeddingFunctionFactory.create_from_dict(init_parameters_data["embedding_function"])
            init_parameters_data["embedding_function"] = embedding_function

        # Load the component
        return default_from_dict(cls, {"type": data["type"], "init_parameters": init_parameters_data})

    def _validate_strategy(self):
        if self.strategy not in ["sentence", "structural", "semantic"]:
            raise ValueError(
                f"Invalid strategy: {self.strategy}. Must be one of 'sentence', 'structural', or 'semantic'."
            )
        if self.strategy == "semantic" and self.embedding_function is None:
            raise ValueError("Embedding function is required for semantic splitting.")

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Splits the input documents based on the specified strategy.

        :param documents: List of Document objects to be split.
        :returns: A dictionary containing the split documents. Those splits will be flattened into a single list.
        """
        match self.strategy:
            case "sentence":
                raw_chunks = [
                    chunk_by_sentence(doc.content, self.max_tokens, self.overlap_tokens, self.token_encoding_model)
                    for doc in documents
                ]
            case "structural":
                raw_chunks = [
                    chunk_by_structure(doc.content, self.max_tokens, self.min_tokens, self.token_encoding_model)
                    for doc in documents
                ]
            case "semantic":
                if self.embedding_function is None:
                    raise ValueError("Embedding function is required for semantic splitting.")
                raw_chunks = [
                    chunk_by_semantic(
                        doc.content,
                        self.max_tokens,
                        self.min_tokens,
                        self.embedding_function,
                        self.token_encoding_model,
                        self.semantic_preferences,
                    )
                    for doc in documents
                ]
            case _:
                raise ValueError(f"Unsupported strategy: {self.strategy}")

        chunks = [
            Document(content=chunk["content"], meta=chunk["metadata"])
            for doc_chunks in raw_chunks
            for chunk in doc_chunks
        ]
        return {"documents": chunks}


def chunk_by_sentence(
    text: Optional[str], max_tokens: int, overlap_tokens: int = 0, model: str = "gpt-4o"
) -> List[Dict[str, Any]]:
    """
    Standalone function that breaks text into chunks of approximately max_tokens with specified overlap.
    Uses paragraph boundaries as natural break points when possible.

    Algorithm:
        1. Split text into paragraphs
        2. Build chunks by adding paragraphs until reaching max_tokens
        3. For oversized paragraphs, split into sentences
        4. When specified, create overlap between consecutive chunks

    Edge cases:
        - Empty text returns empty list
        - Text smaller than max_tokens returns single chunk
        - Single paragraphs exceeding max_tokens are split by sentences
        - Last chunk might be smaller than others

    :param text: The text to chunk
    :param max_tokens: Maximum tokens per chunk (target)
    :param overlap_tokens: Number of tokens to overlap between consecutive chunks. Please note that these tokens
        are not counted in the chunk size.
    :param model: Model name for token counting
    :returns: List of dictionaries with content and metadata (position, tokens)
    """
    # Handle edge cases
    if not text:
        return []

    total_tokens = count_tokens(text, model)
    if total_tokens <= max_tokens:
        return [
            {
                "content": text,
                "metadata": {
                    "position": 0,
                    "tokens": total_tokens,
                },
            }
        ]

    # Split into paragraphs first for more natural chunking
    paragraphs = re.split(r"\n\s*\n", text)

    chunks: List[Dict[str, Any]] = []
    current_paragraphs: List[str] = []
    current_tokens = 0
    position = 0

    for para in paragraphs:
        para_tokens = count_tokens(para, model)

        if para_tokens > max_tokens:
            # Handle oversized paragraph
            if current_paragraphs:
                # First add the current chunk
                chunk_text = "\n\n".join(current_paragraphs)
                chunks.append(
                    {
                        "content": chunk_text,
                        "metadata": {
                            "position": position,
                            "tokens": current_tokens,
                        },
                    }
                )
                position += len(chunk_text) + 2  # +2 for the newlines we'll add between chunks
                current_paragraphs = []
                current_tokens = 0

            # Now handle the large paragraph by splitting into sentences
            sentences = split_sentences(para)
            sentence_chunk: List[str] = []
            sentence_tokens = 0

            for sentence in sentences:
                sent_tokens = count_tokens(sentence, model)

                # If single sentence is too large, include it anyway
                if sent_tokens > max_tokens:
                    if sentence_chunk:
                        # Add current sentences first
                        chunk_text = " ".join(sentence_chunk)
                        chunks.append(
                            {
                                "content": chunk_text,
                                "metadata": {
                                    "position": position,
                                    "tokens": sentence_tokens,
                                },
                            }
                        )
                        position += len(chunk_text) + 1
                        sentence_chunk = []
                        sentence_tokens = 0

                    # Now add the large sentence
                    chunks.append(
                        {
                            "content": sentence,
                            "metadata": {
                                "position": position,
                                "tokens": sent_tokens,
                            },
                        }
                    )
                    position += len(sentence) + 1
                    continue

                # Check if adding this sentence would exceed the limit
                if sentence_tokens + sent_tokens > max_tokens:
                    # Create chunk with sentences so far
                    chunk_text = " ".join(sentence_chunk)
                    chunks.append(
                        {
                            "content": chunk_text,
                            "metadata": {
                                "position": position,
                                "tokens": sentence_tokens,
                            },
                        }
                    )
                    position += len(chunk_text) + 1
                    sentence_chunk = [sentence]
                    sentence_tokens = sent_tokens
                else:
                    # Add sentence to current chunk
                    sentence_chunk.append(sentence)
                    sentence_tokens += sent_tokens

            # Add any remaining sentences
            if sentence_chunk:
                chunk_text = " ".join(sentence_chunk)
                chunks.append(
                    {
                        "content": chunk_text,
                        "metadata": {
                            "position": position,
                            "tokens": sentence_tokens,
                        },
                    }
                )
                position += len(chunk_text) + 2

        elif current_tokens + para_tokens > max_tokens:
            # Current paragraph would exceed the chunk size
            # Create a chunk with the content we have so far
            chunk_text = "\n\n".join(current_paragraphs)
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        "position": position,
                        "tokens": current_tokens,
                    },
                }
            )
            position += len(chunk_text) + 2

            # Start a new chunk with this paragraph
            current_paragraphs = [para]
            current_tokens = para_tokens
        else:
            # Add paragraph to current chunk
            current_paragraphs.append(para)
            current_tokens += para_tokens

    # Add any remaining content
    if current_paragraphs:
        chunk_text = "\n\n".join(current_paragraphs)
        chunks.append(
            {
                "content": chunk_text,
                "metadata": {
                    "position": position,
                    "tokens": current_tokens,
                },
            }
        )

    # Implement overlap if specified
    if overlap_tokens > 0 and len(chunks) > 1:
        chunks_with_overlap = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]["content"]
            curr_chunk = chunks[i]["content"]

            # Get the last overlap_tokens from the previous chunk
            overlap_text = slice_text_by_tokens(prev_chunk, -overlap_tokens, None, model)

            # Simple rules for spacing based on character properties
            needs_space = False

            if overlap_text and curr_chunk:
                # No space needed if either has space at the boundary
                curr_starts_with_space = curr_chunk[0].isspace()

                # Check if the last character is punctuation that DOES require a following space
                # in English and similar languages (period, comma, etc.)
                space_after_punct = ".,:;!?"
                overlap_ends_with_space_punct = overlap_text[-1] in space_after_punct

                # Add space if we have punctuation that needs space after it
                needs_space = overlap_ends_with_space_punct and not curr_starts_with_space

            # Combine with appropriate spacing
            combined_content = overlap_text + (" " if needs_space else "") + curr_chunk

            # Add overlap to beginning of current chunk
            overlapped_chunk = {
                "content": combined_content,
                "metadata": {
                    "position": chunks[i]["metadata"]["position"] - len(overlap_text),
                    "tokens": chunks[i]["metadata"]["tokens"] + overlap_tokens,
                },
            }
            chunks_with_overlap.append(overlapped_chunk)

        chunks = chunks_with_overlap

    return chunks


def chunk_by_structure(
    text: Optional[str],
    max_tokens: int,
    min_tokens: int,
    model: str = "gpt-4o",
) -> List[Dict[str, Any]]:
    """
    Standalone function that chunks content based on Markdown structure with token-based size constraints.
    Preserves heading hierarchy while ensuring reasonable chunk sizes.

    Algorithm:
        1. Identify Markdown headings as primary break points
        2. Extract sections based on heading boundaries
        3. Split oversized sections into smaller chunks while preserving context
        4. Merge small adjacent sections with the same heading when appropriate

    Chunking strategy:
        - Headings define primary chunk boundaries
        - Sections exceeding max_tokens are split by paragraphs, then sentences
        - Sections smaller than min_tokens are merged when they share the same heading
        - Content hierarchy is preserved (H1 > H2 > paragraphs > sentences)

    Edge cases:
        - No headings: falls back to sentence-based chunking
        - Single section exceeding max_tokens: split into smaller chunks
        - End-of-document handling: ensures no tiny chunks are left at the end

    :param text: Markdown text to chunk
    :param max_tokens: Maximum tokens per chunk (target)
    :param min_tokens: Minimum tokens per chunk (unless unavoidable)
    :param model: Model name for token counting
    :returns: List of dictionaries with content and metadata (heading, level, position, tokens)
    """
    # Handle edge cases
    if not text:
        return []

    # Find all headings in the document
    headings_data = extract_markdown_headings(text)
    headings = [("#" * h["level"], h["text"], h["position"]) for h in headings_data]

    # No headings case - use size-based chunking
    if not headings:
        tokens = count_tokens(text, model)
        if tokens <= max_tokens:
            return [
                {
                    "content": text,
                    "metadata": {
                        "tokens": tokens,
                    },
                }
            ]
        return chunk_by_sentence(text, max_tokens, 0, model)

    # Add document end marker
    headings.append((None, None, len(text)))

    # Extract sections based on headings
    sections = []
    for i in range(len(headings) - 1):
        level = len(headings[i][0]) if headings[i][0] else 0
        heading_text = headings[i][1]
        start_pos = headings[i][2]
        end_pos = headings[i + 1][2]

        content = text[start_pos:end_pos].strip()
        if not content:  # Skip empty sections
            continue

        tokens = count_tokens(content, model)
        sections.append(
            {
                "content": content,
                "heading": heading_text,
                "level": level,
                "tokens": tokens,
                "position": start_pos,
            }
        )

    # First pass: Handle sections that exceed max_tokens
    processed_sections = []
    for section in sections:
        if section["tokens"] <= max_tokens:
            processed_sections.append(section)
        else:
            # Split oversized section into smaller chunks
            sub_chunks = chunk_by_sentence(section["content"], max_tokens, 0, model)

            # Add section metadata to each chunk
            for idx, chunk in enumerate(sub_chunks):
                chunk_with_metadata = {
                    "content": chunk["content"],
                    "heading": section["heading"],
                    "level": section["level"],
                    "tokens": chunk["metadata"]["tokens"],
                    "position": section["position"],
                    "is_partial": True,
                    "part": idx + 1,
                    "of_parts": len(sub_chunks),
                }
                processed_sections.append(chunk_with_metadata)

    # Second pass: Merge small adjacent sections
    merged_sections = []
    current = None

    for section in processed_sections:
        # Skip merging for partial sections (already split by size)
        if section.get("is_partial"):
            if current:
                merged_sections.append(current)
                current = None
            merged_sections.append(section)
            continue

        if not current:
            current = section
            continue

        # Should we try to merge?
        should_merge = False

        # Case 1: Current section is very small (less than 50% of min_tokens)
        if current["tokens"] < min_tokens * 0.5:
            should_merge = True

        # Case 2: Both sections are under min_tokens
        elif current["tokens"] < min_tokens and section["tokens"] < min_tokens:
            should_merge = True

        # Case 3: Current is under min and next is a subsection (higher level number)
        elif current["tokens"] < min_tokens and section["level"] > current["level"]:
            should_merge = True

        # Check if merging is feasible size-wise
        if should_merge:
            combined_tokens = current["tokens"] + count_tokens("\n\n", model) + section["tokens"]

            if combined_tokens <= max_tokens * 1.1:  # Allow 10% overflow
                # Merge the sections
                current["content"] += "\n\n" + section["content"]
                current["tokens"] = combined_tokens
                # If we're merging in promotional content (level 2+ with tiny content),
                # don't change the primary heading
                if not (section["level"] > 1 and section["tokens"] < 100):
                    current["heading"] += " + " + section["heading"]
                continue

        # If we get here, we couldn't or shouldn't merge
        merged_sections.append(current)
        current = section

    # Don't forget the last section
    if current:
        merged_sections.append(current)

    # Convert sections to final chunk format
    chunks = []
    for section in merged_sections:
        chunks.append(
            {
                "content": section["content"],
                "metadata": {
                    "heading": section["heading"],
                    "level": section["level"],
                    "position": section["position"],
                    "tokens": section["tokens"],
                    **({"part": section["part"], "of_parts": section["of_parts"]} if section.get("is_partial") else {}),
                },
            }
        )

    return chunks


def chunk_by_semantic(
    text: Optional[str],
    max_tokens: int,
    min_tokens: int,
    embedding_function: EmbeddingFunction,
    model: str = "gpt-4o",
    params: Optional[SemanticPreferenceConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Standalone function that creates semantically coherent chunks based on embedding similarity between paragraphs.
    Balances semantic boundaries with token size constraints.

    Algorithm:
        1. Split text into paragraphs (fundamental semantic units)
        2. Generate embeddings for each paragraph using embedding_model_service
        3. Calculate similarity scores between adjacent paragraphs
        4. Create chunks by breaking at semantic boundaries while respecting token limits
        5. Look ahead to avoid creating tiny chunks at the end

    Break point selection:
        1. Once min_tokens is reached, evaluate each potential break point
        2. Score breaks based on semantic dissimilarity and heading presence
        3. Take breaks that exceed the threshold, adjusting threshold as chunk grows
        4. Look ahead to avoid creating chunks smaller than min_chunk_ratio * min_tokens

    Tuning guidance:
        - For more aggressive semantic splitting: lower strong_break_threshold
        - For larger chunks: increase dynamic_threshold_base
        - For stronger heading-based breaks: increase heading_bonus
        - For more aggressive merging of small chunks: increase min_chunk_ratio

    :param text: Markdown text to chunk
    :param max_tokens: Maximum tokens per chunk (hard constraint)
    :param min_tokens: Minimum tokens per chunk (soft constraint)
    :param embedding_function: Embedding function to use for semantic splitting
    :param model: Model name for token counting
    :param params: Configurable semantic preferences
    :returns: List of dictionaries with content and metadata including:
        - tokens: Token count for the chunk
        - semantic: Flag indicating semantic chunking
        - paragraphs: Number of paragraphs in the chunk
        - coherence: Average similarity between paragraphs within the chunk
        - next_similarity: Similarity score with the following chunk
    """
    # Set default parameters if not provided
    if params is None:
        params = SemanticPreferenceConfig()

    # Handle edge cases
    if not text:
        return []

    total_tokens = count_tokens(text, model)
    if total_tokens <= max_tokens:
        return [{"content": text, "metadata": {"tokens": total_tokens, "semantic": True}}]

    # Step 1: Split text into semantic units (paragraphs)
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p for p in paragraphs if p.strip()]

    # Identify heading paragraphs
    heading_indices = []
    for i, para in enumerate(paragraphs):
        if re.match(r"^#{1,6}\s+.+$", para.strip().split("\n")[0]):
            heading_indices.append(i)

    # Step 2: Calculate embeddings for each paragraph
    embeddings = np.array(embedding_function(paragraphs))
    unit_tokens = [count_tokens(para, model) for para in paragraphs]

    # Step 3: Calculate semantic similarity between adjacent paragraphs
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(sim)

    # Step 4: Look ahead to plan chunks optimally
    chunks = []
    start_idx = 0

    while start_idx < len(paragraphs):
        # Step 4a: Find the best end point for this chunk
        best_end_idx = _find_best_chunk_end(
            start_idx,
            paragraphs,
            unit_tokens,
            similarities,
            heading_indices,
            min_tokens,
            max_tokens,
            params,
        )

        # Step 4b: Look ahead to check if the next chunk would be too small
        next_chunk_start = best_end_idx + 1
        if next_chunk_start < len(paragraphs):
            # Calculate how many paragraphs & tokens would be in the next chunk
            remaining_tokens = 0

            for i in range(next_chunk_start, len(paragraphs)):
                if i > next_chunk_start:
                    remaining_tokens += 2  # For paragraph separators
                remaining_tokens += unit_tokens[i]

            # If the remaining content is too small to make a good chunk
            # AND it would fit in the current chunk, extend the current chunk
            if (
                remaining_tokens < min_tokens * params.min_chunk_ratio
                and remaining_tokens + sum(unit_tokens[start_idx : best_end_idx + 1]) + 2 <= max_tokens
            ):

                # Check similarity with what would be the next chunk
                should_extend = True
                if best_end_idx < len(similarities):
                    # If the similarity is very low, don't extend
                    if 1 - similarities[best_end_idx] > params.exceptional_break_threshold:
                        should_extend = False

                if should_extend:
                    best_end_idx = len(paragraphs) - 1

        # Create the chunk
        chunk_paras = paragraphs[start_idx : best_end_idx + 1]
        chunk_text = "\n\n".join(chunk_paras)
        chunk_tokens = sum(unit_tokens[start_idx : best_end_idx + 1])
        if len(chunk_paras) > 1:
            chunk_tokens += 2 * (len(chunk_paras) - 1)  # Account for "\n\n" separators

        # Calculate semantic coherence score for this chunk
        coherence_score = None
        if best_end_idx > start_idx and start_idx < len(similarities):
            chunk_similarities = similarities[start_idx : min(best_end_idx, len(similarities))]
            if chunk_similarities:
                coherence_score = sum(chunk_similarities) / len(chunk_similarities)

        # Add chunk
        chunk_metadata: Dict[str, Any] = {
            "tokens": chunk_tokens,
            "semantic": True,
            "paragraphs": best_end_idx - start_idx + 1,
        }

        if coherence_score is not None:
            chunk_metadata["coherence"] = coherence_score

        # Add boundary information if applicable
        if best_end_idx < len(similarities):
            chunk_metadata["next_similarity"] = similarities[best_end_idx]

        chunks.append({"content": chunk_text, "metadata": chunk_metadata})

        # Move to the next chunk
        start_idx = best_end_idx + 1

    return chunks


def _find_best_chunk_end(
    start_idx: int,
    paragraphs: List[str],
    token_counts: List[int],
    similarities: List[float],
    heading_indices: List[int],
    min_tokens: int,
    max_tokens: int,
    params: SemanticPreferenceConfig,
) -> int:
    """
    Find the optimal end point for a chunk starting at start_idx.

    This helper function evaluates potential break points to find the best semantic boundary that also respects token
    constraints. Key criteria:
        1. Must have at least min_tokens of content
        2. Must not exceed max_tokens
        3. Prefers breaking at strong semantic boundaries (low similarity)
        4. Prefers breaking at headings
        5. Gets less strict about semantic requirements as chunk approaches max_tokens

    Selection process:
        1. Track the best semantic break seen after reaching min_tokens
        2. For exceptional semantic breaks, return immediately
        3. For strong breaks after reaching the midpoint, accept more readily
        4. Default to finishing at max_tokens or document end

    :param start_idx: Starting paragraph index.
    :param paragraphs: List of all paragraphs in document.
    :param token_counts: Pre-computed token count for each paragraph.
    :param similarities: Similarity scores between adjacent paragraphs.
    :param heading_indices: Indices of paragraphs that start with headings.
    :param min_tokens: Minimum tokens for a valid chunk.
    :param max_tokens: Maximum tokens allowed per chunk.
    :param params: Dictionary of configurable scoring parameters.
    :returns: Index of the paragraph where the chunk should end.
    """
    # Handle end of document case
    if start_idx >= len(paragraphs) - 1:
        return len(paragraphs) - 1

    # Initialize tracking variables
    current_tokens = token_counts[start_idx]
    best_semantic_break_idx = -1
    best_semantic_break_score = -float("inf")

    # Evaluate each possible end point
    for end_idx in range(start_idx, len(paragraphs) - 1):
        # Add next paragraph's tokens with separator
        if end_idx > start_idx:
            current_tokens += 2 + token_counts[end_idx]  # +2 for "\n\n"

        # If we'd exceed max_tokens with the next paragraph, stop here
        next_tokens = current_tokens + 2 + token_counts[end_idx + 1] if end_idx + 1 < len(token_counts) else 0
        if next_tokens > max_tokens and current_tokens >= min_tokens:
            return end_idx

        # Once we reach min_tokens, start evaluating semantic breaks
        if current_tokens >= min_tokens:
            # Consider headings as excellent break points
            heading_bonus = params.heading_bonus if end_idx + 1 in heading_indices else 0

            if end_idx < len(similarities):
                # Calculate semantic break score - higher means better break point
                semantic_score = 1.0 - similarities[end_idx]

                # Adjust threshold based on how close we are to max_tokens
                size_ratio = current_tokens / max_tokens  # How close we are to max_tokens
                dynamic_threshold = params.dynamic_threshold_base * (1.0 - size_ratio)

                # If this is a strong semantic break
                if semantic_score > params.strong_break_threshold or heading_bonus > 0:
                    break_score = semantic_score + heading_bonus

                    # Track the best semantic break
                    if break_score > best_semantic_break_score:
                        best_semantic_break_score = break_score
                        best_semantic_break_idx = end_idx

                        # If this is an exceptionally good break, use it
                        if break_score > params.exceptional_break_threshold:
                            return end_idx

                # As we get closer to max_tokens, accept weaker breaks
                if semantic_score > dynamic_threshold and current_tokens > (min_tokens + max_tokens) / 2:
                    return end_idx

    # If we've looked at all paragraphs:

    # If we found a good semantic break after min_tokens, use it
    if best_semantic_break_idx >= 0:
        return best_semantic_break_idx

    # If we have enough content and are near the end, take all remaining paragraphs
    if current_tokens >= min_tokens:
        # Check if adding the last paragraph would exceed max_tokens
        if current_tokens + token_counts[-1] + 2 <= max_tokens:
            return len(paragraphs) - 1
        return len(paragraphs) - 2

    # Otherwise, return what we have
    return len(paragraphs) - 1
