"""
Basic unit tests for RAG Service core logic without heavy dependencies.

Tests the core functionality that doesn't require external services.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

# Mock the heavy dependencies before importing
import sys
from unittest.mock import MagicMock

# Mock modules that have heavy dependencies
sys.modules['faiss'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['llama_index'] = MagicMock()
sys.modules['llama_index.core'] = MagicMock()
sys.modules['llama_index.core.node_parser'] = MagicMock()
sys.modules['llama_index.readers.file'] = MagicMock()
sys.modules['llama_index.core.schema'] = MagicMock()

# Mock config settings
mock_settings = MagicMock()
mock_settings.llm.max_tokens = 1024
mock_settings.llm.temperature = 0.7
mock_settings.vector_store.embedding_dimension = 768
mock_settings.embedding.batch_size = 32
mock_settings.llm.api_key = "test-api-key"
sys.modules['config'] = MagicMock()
sys.modules['config.settings'] = MagicMock()
sys.modules['config.settings'].settings = mock_settings

from src.services.rag_service import GeminiAPIClient, RAGServiceError
from src.models.data_models import DocumentChunk, TokenUsage


class TestGeminiAPIClientBasic:
    """Basic test cases for GeminiAPIClient without external dependencies."""
    
    @pytest.fixture
    def gemini_client(self):
        """Create GeminiAPIClient instance for testing."""
        return GeminiAPIClient(api_key="test-api-key", model_name="gemini-pro")
    
    def test_init(self, gemini_client):
        """Test GeminiAPIClient initialization."""
        assert gemini_client.api_key == "test-api-key"
        assert gemini_client.model_name == "gemini-pro"
        assert gemini_client.base_url == "https://generativelanguage.googleapis.com/v1beta"
    
    def test_construct_prompt_with_context(self, gemini_client):
        """Test prompt construction with context."""
        prompt = gemini_client._construct_prompt(
            query="What is the capital of France?",
            context="France is a country in Europe. Paris is its capital city."
        )
        
        assert "France is a country in Europe" in prompt
        assert "What is the capital of France?" in prompt
        assert "Context:" in prompt
        assert "Answer:" in prompt
    
    def test_construct_prompt_without_context(self, gemini_client):
        """Test prompt construction without context."""
        prompt = gemini_client._construct_prompt(
            query="What is the capital of France?"
        )
        
        assert "What is the capital of France?" in prompt
        assert "Context:" not in prompt
        assert "Answer:" in prompt
    
    def test_estimate_tokens(self, gemini_client):
        """Test token estimation."""
        text = "This is a test sentence with eight words."
        tokens = gemini_client._estimate_tokens(text)
        
        # Should be roughly 8 * 1.33 ≈ 10-11 tokens
        assert 10 <= tokens <= 12
        
        # Test empty text
        assert gemini_client._estimate_tokens("") == 1
        
        # Test single word
        assert gemini_client._estimate_tokens("word") >= 1
    
    def test_parse_response_success(self, gemini_client):
        """Test successful response parsing."""
        mock_response_data = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": "This is a test response from Gemini API."
                    }]
                }
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 8,
                "totalTokenCount": 18
            }
        }
        
        mock_response = Mock()
        mock_response.json.return_value = mock_response_data
        
        response_text, token_usage = gemini_client._parse_response(
            mock_response, 
            "test prompt"
        )
        
        assert response_text == "This is a test response from Gemini API."
        assert token_usage.prompt_tokens == 10
        assert token_usage.completion_tokens == 8
        assert token_usage.total_tokens == 18
    
    def test_parse_response_no_candidates(self, gemini_client):
        """Test response parsing with no candidates."""
        mock_response = Mock()
        mock_response.json.return_value = {"candidates": []}
        
        with pytest.raises(RAGServiceError, match="No candidates in API response"):
            gemini_client._parse_response(mock_response, "test prompt")
    
    def test_parse_response_empty_text(self, gemini_client):
        """Test response parsing with empty text."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": ""
                    }]
                }
            }]
        }
        
        with pytest.raises(RAGServiceError, match="Empty response from API"):
            gemini_client._parse_response(mock_response, "test prompt")
    
    def test_parse_response_missing_usage_metadata(self, gemini_client):
        """Test response parsing without usage metadata."""
        mock_response_data = {
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": "Response without usage metadata."
                    }]
                }
            }]
            # No usageMetadata
        }
        
        mock_response = Mock()
        mock_response.json.return_value = mock_response_data
        
        response_text, token_usage = gemini_client._parse_response(
            mock_response, 
            "test prompt with five words"
        )
        
        assert response_text == "Response without usage metadata."
        # Should use estimation
        assert token_usage.prompt_tokens > 0
        assert token_usage.completion_tokens > 0
        assert token_usage.total_tokens > 0


class TestDocumentChunkHelpers:
    """Test helper functions for document chunk processing."""
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample document chunks for testing."""
        return [
            DocumentChunk(
                chunk_id="chunk1",
                document_id="doc1",
                content="This is the first test chunk about artificial intelligence.",
                metadata={"file_name": "ai_doc.pdf", "similarity_score": 0.9},
                chunk_index=0,
                token_count=10
            ),
            DocumentChunk(
                chunk_id="chunk2",
                document_id="doc2",
                content="This is the second test chunk about machine learning.",
                metadata={"file_name": "ml_doc.pdf", "similarity_score": 0.8},
                chunk_index=0,
                token_count=10
            ),
            DocumentChunk(
                chunk_id="chunk3",
                document_id="doc1",
                content="This is the third test chunk about deep learning.",
                metadata={"file_name": "ai_doc.pdf", "similarity_score": 0.7},
                chunk_index=1,
                token_count=10
            )
        ]
    
    def test_build_context_string(self, sample_chunks):
        """Test context string building."""
        # Import here to avoid dependency issues
        from src.services.rag_service import RAGService
        
        # Create a mock RAG service just for testing this method
        mock_vector_storage = Mock()
        mock_embedding_service = Mock()
        
        rag_service = RAGService(
            vector_storage=mock_vector_storage,
            embedding_service=mock_embedding_service,
            gemini_api_key="test-key"
        )
        
        context = rag_service._build_context_string(sample_chunks)
        
        assert "artificial intelligence" in context
        assert "machine learning" in context
        assert "deep learning" in context
        assert "[Context 1" in context
        assert "[Context 2" in context
        assert "ai_doc.pdf" in context
        assert "ml_doc.pdf" in context
        assert "relevance: 0.90" in context
        assert "relevance: 0.80" in context
    
    def test_build_context_string_empty(self):
        """Test context string building with empty chunks."""
        from src.services.rag_service import RAGService
        
        mock_vector_storage = Mock()
        mock_embedding_service = Mock()
        
        rag_service = RAGService(
            vector_storage=mock_vector_storage,
            embedding_service=mock_embedding_service,
            gemini_api_key="test-key"
        )
        
        context = rag_service._build_context_string([])
        assert context == ""
    
    def test_calculate_confidence_score(self, sample_chunks):
        """Test confidence score calculation."""
        from src.services.rag_service import RAGService
        
        mock_vector_storage = Mock()
        mock_embedding_service = Mock()
        
        rag_service = RAGService(
            vector_storage=mock_vector_storage,
            embedding_service=mock_embedding_service,
            gemini_api_key="test-key",
            max_context_chunks=5
        )
        
        confidence = rag_service._calculate_confidence_score(sample_chunks, "test query")
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably high with good similarity scores
    
    def test_calculate_confidence_score_no_chunks(self):
        """Test confidence score calculation with no chunks."""
        from src.services.rag_service import RAGService
        
        mock_vector_storage = Mock()
        mock_embedding_service = Mock()
        
        rag_service = RAGService(
            vector_storage=mock_vector_storage,
            embedding_service=mock_embedding_service,
            gemini_api_key="test-key"
        )
        
        confidence = rag_service._calculate_confidence_score([], "test query")
        assert confidence == 0.0
    
    def test_filter_and_rank_chunks(self, sample_chunks):
        """Test chunk filtering and ranking."""
        from src.services.rag_service import RAGService
        
        mock_vector_storage = Mock()
        mock_embedding_service = Mock()
        
        rag_service = RAGService(
            vector_storage=mock_vector_storage,
            embedding_service=mock_embedding_service,
            gemini_api_key="test-key"
        )
        
        filtered_chunks = rag_service._filter_and_rank_chunks(sample_chunks, limit=2)
        
        # Should return top 2 chunks by similarity
        assert len(filtered_chunks) == 2
        assert filtered_chunks[0].metadata["similarity_score"] >= filtered_chunks[1].metadata["similarity_score"]
    
    def test_filter_and_rank_chunks_diversity(self, sample_chunks):
        """Test chunk filtering with diversity consideration."""
        from src.services.rag_service import RAGService
        
        # All chunks have same high similarity but different documents
        for chunk in sample_chunks:
            chunk.metadata["similarity_score"] = 0.9
        
        mock_vector_storage = Mock()
        mock_embedding_service = Mock()
        
        rag_service = RAGService(
            vector_storage=mock_vector_storage,
            embedding_service=mock_embedding_service,
            gemini_api_key="test-key"
        )
        
        filtered_chunks = rag_service._filter_and_rank_chunks(sample_chunks, limit=3)
        
        # Should prefer diversity across documents
        document_ids = [chunk.document_id for chunk in filtered_chunks]
        unique_docs = len(set(document_ids))
        assert unique_docs >= 2  # Should have chunks from multiple documents


if __name__ == "__main__":
    pytest.main([__file__])