import unittest
from unittest.mock import patch, MagicMock

from haystack.dataclasses import Document

from ravelin.embedding_functions import RegisteredEmbeddingFunction
from ravelin.components.rankers.embedding_similarity_ranker import EmbeddingSimilarityRanker


class TestEmbeddingSimilarityRanker(unittest.TestCase):
    def setUp(self):
        # Mock the embedding function
        self.mock_embedding_function = MagicMock(spec=RegisteredEmbeddingFunction)
        self.mock_embedding_function.to_dict.return_value = {"type": "openai", "name": "test_embedding"}

        # Sample documents
        self.doc1 = Document(id="1", content="Document 1")
        self.doc2 = Document(id="2", content="Document 2")
        self.doc3 = Document(id="3", content="Document 3")

        # Sample query
        self.query = "test query"

    def test_initialization(self):
        """Test the initialization of the ranker with various parameters"""
        # Arrange & Act
        ranker = EmbeddingSimilarityRanker(
            embedding_function=self.mock_embedding_function,
            top_k=5,
            score_threshold=0.7,
            recompute_embeddings=True,
        )

        # Assert
        self.assertEqual(self.mock_embedding_function, ranker.embedding_function)
        self.assertEqual(5, ranker.top_k)
        self.assertEqual(0.7, ranker.score_threshold)
        self.assertTrue(ranker.recompute_embeddings)

    def test_document_ranking(self):
        """Test that documents are correctly ranked by similarity to query"""

        # Arrange
        def mock_embed_func(texts):
            results = []
            for text in texts:
                if text == self.query:
                    results.append([1.0, 0.0, 0.0])  # Query embedding
                elif text == "Document 1":
                    results.append([0.8, 0.1, 0.1])  # Cosine similarity ≈ 0.98
                elif text == "Document 2":
                    results.append([0.5, 0.5, 0.5])  # Cosine similarity ≈ 0.58
                elif text == "Document 3":
                    results.append([0.2, 0.7, 0.7])  # Cosine similarity ≈ 0.20
                else:
                    results.append([0.0, 0.0, 0.0])
            return results

        self.mock_embedding_function.side_effect = mock_embed_func

        ranker = EmbeddingSimilarityRanker(embedding_function=self.mock_embedding_function)

        # Act
        result = ranker.run(query=self.query, documents=[self.doc1, self.doc2, self.doc3])
        ranked_docs = result["documents"]

        # Assert
        self.assertEqual(3, len(ranked_docs))
        self.assertEqual("1", ranked_docs[0].id)  # Most similar
        self.assertEqual("2", ranked_docs[1].id)  # Second most similar
        self.assertEqual("3", ranked_docs[2].id)  # Least similar

        # Check scores are within expected range
        self.assertAlmostEqual(0.98, ranked_docs[0].score, places=2)
        self.assertAlmostEqual(0.58, ranked_docs[1].score, places=2)
        self.assertAlmostEqual(0.20, ranked_docs[2].score, places=2)

    def test_top_k_parameter(self):
        """Test limiting results with top_k parameter"""

        # Arrange
        def mock_embed_func(texts):
            results = []
            for text in texts:
                if text == self.query:
                    results.append([1.0, 0.0, 0.0])
                else:
                    results.append([0.5, 0.5, 0.5])
            return results

        self.mock_embedding_function.side_effect = mock_embed_func

        ranker = EmbeddingSimilarityRanker(embedding_function=self.mock_embedding_function)

        # Act - test with run parameter
        result = ranker.run(query=self.query, documents=[self.doc1, self.doc2, self.doc3], top_k=2)

        # Assert
        self.assertEqual(2, len(result["documents"]))

        # Act - test with constructor parameter
        ranker_with_top_k = EmbeddingSimilarityRanker(embedding_function=self.mock_embedding_function, top_k=1)
        result = ranker_with_top_k.run(query=self.query, documents=[self.doc1, self.doc2, self.doc3])

        # Assert
        self.assertEqual(1, len(result["documents"]))

        # Act & Assert - test invalid top_k
        with self.assertRaises(ValueError):
            ranker.run(query=self.query, documents=[self.doc1, self.doc2, self.doc3], top_k=0)

    def test_score_threshold_parameter(self):
        """Test filtering results with score_threshold parameter"""

        # Arrange
        def mock_embed_func(texts):
            results = []
            for text in texts:
                if text == self.query:
                    results.append([1.0, 0.0, 0.0])
                elif text == "Document 1":
                    results.append([0.8, 0.1, 0.1])  # Cosine similarity ≈ 0.98
                elif text == "Document 2":
                    results.append([0.5, 0.5, 0.5])  # Cosine similarity ≈ 0.58
                elif text == "Document 3":
                    results.append([0.2, 0.7, 0.7])  # Cosine similarity ≈ 0.20
                else:
                    results.append([0.0, 0.0, 0.0])
            return results

        self.mock_embedding_function.side_effect = mock_embed_func

        ranker = EmbeddingSimilarityRanker(embedding_function=self.mock_embedding_function)

        # Act - test with run parameter
        result = ranker.run(query=self.query, documents=[self.doc1, self.doc2, self.doc3], score_threshold=0.5)

        # Assert
        self.assertEqual(2, len(result["documents"]))
        self.assertEqual("1", result["documents"][0].id)
        self.assertEqual("2", result["documents"][1].id)

        # Act - test with constructor parameter
        ranker_with_threshold = EmbeddingSimilarityRanker(
            embedding_function=self.mock_embedding_function, score_threshold=0.9
        )
        result = ranker_with_threshold.run(query=self.query, documents=[self.doc1, self.doc2, self.doc3])

        # Assert
        self.assertEqual(1, len(result["documents"]))
        self.assertEqual("1", result["documents"][0].id)

    def test_recompute_embeddings_parameter(self):
        """Test recomputation of embeddings based on parameter"""

        # Arrange
        def mock_embed_func(texts):
            return [[0.5, 0.5, 0.5] for _ in texts]

        self.mock_embedding_function.side_effect = mock_embed_func

        doc1_with_embedding = Document(id="1", content="Document 1", embedding=[0.8, 0.1, 0.1])
        doc2_with_embedding = Document(id="2", content="Document 2", embedding=[0.5, 0.5, 0.5])

        ranker = EmbeddingSimilarityRanker(embedding_function=self.mock_embedding_function)

        # Act - test without recomputing (default behavior)
        self.mock_embedding_function.reset_mock()
        ranker.run(query=self.query, documents=[doc1_with_embedding, doc2_with_embedding])

        # Assert - embedding function only called for query
        self.assertEqual(1, self.mock_embedding_function.call_count)
        self.mock_embedding_function.assert_called_once_with([self.query])

        # Act - test with recomputing
        self.mock_embedding_function.reset_mock()
        ranker.run(query=self.query, documents=[doc1_with_embedding, doc2_with_embedding], recompute_embeddings=True)

        # Assert - embedding function called for query and both documents
        self.assertEqual(2, self.mock_embedding_function.call_count)  # One call for query, one for docs batch
        # First call for query
        self.mock_embedding_function.assert_any_call([self.query])
        # Second call for documents batch
        self.mock_embedding_function.assert_any_call(["Document 1", "Document 2"])

    def test_empty_documents(self):
        """Test handling of empty document list"""
        # Arrange
        ranker = EmbeddingSimilarityRanker(embedding_function=self.mock_embedding_function)

        # Act
        result = ranker.run(query=self.query, documents=[])

        # Assert
        self.assertEqual([], result["documents"])

    def test_mixed_document_embeddings(self):
        """Test with mix of documents with and without embeddings"""

        # Arrange
        def mock_embed_func(texts):
            results = []
            for text in texts:
                if text == self.query:
                    results.append([1.0, 0.0, 0.0])
                elif text == "Document 2":
                    results.append([0.5, 0.5, 0.5])
                else:
                    results.append([0.0, 0.0, 0.0])
            return results

        self.mock_embedding_function.side_effect = mock_embed_func

        # Doc with embedding, no content
        doc1 = Document(id="1", embedding=[0.8, 0.1, 0.1])
        # Doc with content, needs embedding
        doc2 = Document(id="2", content="Document 2")
        # Doc without content or embedding (will be skipped)
        doc3 = Document(id="3")

        ranker = EmbeddingSimilarityRanker(embedding_function=self.mock_embedding_function)

        # Act
        self.mock_embedding_function.reset_mock()
        result = ranker.run(query=self.query, documents=[doc1, doc2, doc3])

        # Assert
        ranked_docs = result["documents"]
        self.assertEqual(2, len(ranked_docs))
        self.assertEqual("1", ranked_docs[0].id)  # Should be first due to higher similarity
        self.assertEqual("2", ranked_docs[1].id)  # Should be second

        # Called for query and doc2 only (doc1 has embedding, doc3 has no content)
        self.assertEqual(2, self.mock_embedding_function.call_count)
        # First call for query
        self.mock_embedding_function.assert_any_call([self.query])
        # Second call for documents batch
        self.mock_embedding_function.assert_any_call(["Document 2"])

    @patch("ravelin.embedding_functions.EmbeddingFunctionFactory.create_from_dict")
    def test_serialization_deserialization(self, mock_create_from_dict):
        """Test to_dict and from_dict methods"""
        # Arrange
        mock_create_from_dict.return_value = self.mock_embedding_function

        original_ranker = EmbeddingSimilarityRanker(
            embedding_function=self.mock_embedding_function,
            top_k=5,
            score_threshold=0.7,
            recompute_embeddings=True,
        )

        # Act
        serialized = original_ranker.to_dict()

        # Assert serialization
        self.assertEqual("EmbeddingSimilarityRanker", serialized["type"].split(".")[-1])
        self.assertEqual(
            self.mock_embedding_function.to_dict.return_value, serialized["init_parameters"]["embedding_function"]
        )
        self.assertEqual(5, serialized["init_parameters"]["top_k"])
        self.assertEqual(0.7, serialized["init_parameters"]["score_threshold"])
        self.assertTrue(serialized["init_parameters"]["recompute_embeddings"])

        # Act - deserialization
        EmbeddingSimilarityRanker.from_dict(serialized)

        # Assert deserialization
        mock_create_from_dict.assert_called_once_with(self.mock_embedding_function.to_dict.return_value)


if __name__ == "__main__":
    unittest.main()
