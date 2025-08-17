"""
Unit tests for RAG Service with mocked external dependencies.

Tests the core RAG functionality including query processing, retrieval,
response generation, and error handling scenarios.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from typing import List, Dict, Any

from src.services.rag_service import (
    RAGService, 
    GeminiAPIClient, 
    RAGServiceError,
    create_rag_service
)
from src.models.data_models import DocumentChunk, RAGResponse, TokenUsage
from src.services.vector_storage import FAISSVectorStorage, VectorStorageError
from src.services.document_chunking import EmbeddingService, DocumentChunkingError


class TestGeminiAPIClient:
    """Test cases for GeminiAPIClient."""
    
    @pytest.fixture
    def gemini_client(self):
        """Create GeminiAPIClient instance for testing."""
        return GeminiAPIClient(api_key="test-api-key", model_name="gemini-pro")
    
    @pytest.fixture
    def mock_response_data(self):
        """Mock Gemini API response data."""
        return {
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
    
    @pytest.mark.asyncio
    async def test_generate_response_success(self, gemini_client, mock_response_data):
        """Test successful response generation."""
        with patch.object(gemini_client, '_make_request_with_retry') as mock_request:
            mock_response = Mock()
            mock_response.json.return_value = mock_response_data
            mock_request.return_value = mock_response
            
            response_text, token_usage = await gemini_client.generate_response(
                prompt="Test question",
                max_tokens=100,
                temperature=0.7
            )
            
            assert response_text == "This is a test response from Gemini API."
            assert token_usage.prompt_tokens == 10
            assert token_usage.completion_tokens == 8
            assert token_usage.total_tokens == 18
    
    @pytest.mark.asyncio
    async def test_generate_response_with_context(self, gemini_client, mock_response_data):
        """Test response generation with context."""
        with patch.object(gemini_client, '_make_request_with_retry') as mock_request:
            mock_response = Mock()
            mock_response.json.return_value = mock_response_data
            mock_request.return_value = mock_response
            
            response_text, token_usage = await gemini_client.generate_response(
                prompt="Test question",
                context="Test context information"
            )
            
            # Verify that context was included in the prompt
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            payload = call_args[0][2]  # Third argument is payload
            prompt_text = payload["contents"][0]["parts"][0]["text"]
            
            assert "Test context information" in prompt_text
            assert "Test question" in prompt_text
    
    def test_construct_prompt_with_context(self, gemini_client):
        """Test prompt construction with context."""
        prompt = gemini_client._construct_prompt(
            query="What is the capital of France?",
            context="France is a country in Europe. Paris is its capital city."
        )
        
        assert "France is a country in Europe" in prompt
        assert "What is the capital of France?" in prompt
        assert "Context:" in prompt
    
    def test_construct_prompt_without_context(self, gemini_client):
        """Test prompt construction without context."""
        prompt = gemini_client._construct_prompt(
            query="What is the capital of France?"
        )
        
        assert "What is the capital of France?" in prompt
        assert "Context:" not in prompt
    
    @pytest.mark.asyncio
    async def test_make_request_with_retry_success(self, gemini_client):
        """Test successful API request."""
        with patch.object(gemini_client.client, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            
            result = await gemini_client._make_request_with_retry(
                url="test-url",
                headers={"test": "header"},
                payload={"test": "data"}
            )
            
            assert result == mock_response
            mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_make_request_with_retry_rate_limit(self, gemini_client):
        """Test retry logic for rate limiting."""
        with patch.object(gemini_client.client, 'post') as mock_post:
            # First call returns 429, second call succeeds
            mock_response_429 = Mock()
            mock_response_429.status_code = 429
            mock_response_200 = Mock()
            mock_response_200.status_code = 200
            
            mock_post.side_effect = [mock_response_429, mock_response_200]
            
            with patch('asyncio.sleep') as mock_sleep:
                result = await gemini_client._make_request_with_retry(
                    url="test-url",
                    headers={"test": "header"},
                    payload={"test": "data"},
                    max_retries=1
                )
                
                assert result == mock_response_200
                assert mock_post.call_count == 2
                mock_sleep.assert_called_once_with(1)  # 2^0 = 1
    
    @pytest.mark.asyncio
    async def test_make_request_with_retry_max_retries_exceeded(self, gemini_client):
        """Test failure after max retries exceeded."""
        with patch.object(gemini_client.client, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = Exception("Server error")
            mock_post.return_value = mock_response
            
            with patch('asyncio.sleep'):
                with pytest.raises(RAGServiceError, match="API request failed after"):
                    await gemini_client._make_request_with_retry(
                        url="test-url",
                        headers={"test": "header"},
                        payload={"test": "data"},
                        max_retries=1
                    )
    
    def test_parse_response_success(self, gemini_client, mock_response_data):
        """Test successful response parsing."""
        mock_response = Mock()
        mock_response.json.return_value = mock_response_data
        
        response_text, token_usage = gemini_client._parse_response(
            mock_response, 
            "test prompt"
        )
        
        assert response_text == "This is a test response from Gemini API."
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
    
    def test_estimate_tokens(self, gemini_client):
        """Test token estimation."""
        text = "This is a test sentence with eight words."
        tokens = gemini_client._estimate_tokens(text)
        
        # Should be roughly 8 * 1.33 ≈ 10-11 tokens
        assert 10 <= tokens <= 12


class TestRAGService:
    """Test cases for RAGService."""
    
    @pytest.fixture
    def mock_vector_storage(self):
        """Create mock vector storage."""
        mock_storage = Mock(spec=FAISSVectorStorage)
        mock_storage.get_stats.return_value = {
            "total_vectors": 100,
            "index_type": "HNSW",
            "dimension": 768
        }
        return mock_storage
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock embedding service."""
        mock_service = Mock(spec=EmbeddingService)
        mock_service.model_name = "test-model"
        mock_service.generate_query_embedding.return_value = [0.1] * 768
        return mock_service
    
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
    
    @pytest.fixture
    def rag_service(self, mock_vector_storage, mock_embedding_service):
        """Create RAGService instance for testing."""
        return RAGService(
            vector_storage=mock_vector_storage,
            embedding_service=mock_embedding_service,
            gemini_api_key="test-api-key",
            gemini_model="gemini-pro",
            max_context_chunks=5,
            similarity_threshold=0.7
        )
    
    @pytest.mark.asyncio
    async def test_process_query_success(self, rag_service, sample_chunks):
        """Test successful query processing."""
        # Mock the retrieve_relevant_chunks method
        with patch.object(rag_service, 'retrieve_relevant_chunks') as mock_retrieve:
            mock_retrieve.return_value = sample_chunks
            
            # Mock the generate_response method
            with patch.object(rag_service, 'generate_response') as mock_generate:
                mock_token_usage = TokenUsage(
                    prompt_tokens=50,
                    completion_tokens=30,
                    total_tokens=80
                )
                mock_generate.return_value = ("This is a test answer.", mock_token_usage)
                
                # Process query
                response = await rag_service.process_query(
                    query="What is artificial intelligence?",
                    context_limit=3,
                    include_sources=True
                )
                
                # Verify response
                assert isinstance(response, RAGResponse)
                assert response.query == "What is artificial intelligence?"
                assert response.answer == "This is a test answer."
                assert len(response.source_chunks) <= 3
                assert response.confidence_score > 0.0
                assert response.processing_time_ms > 0
                assert response.token_usage.total_tokens == 80
    
    @pytest.mark.asyncio
    async def test_process_query_empty_query(self, rag_service):
        """Test query processing with empty query."""
        with pytest.raises(RAGServiceError, match="Query cannot be empty"):
            await rag_service.process_query("")
    
    @pytest.mark.asyncio
    async def test_process_query_no_sources(self, rag_service, sample_chunks):
        """Test query processing without including sources."""
        with patch.object(rag_service, 'retrieve_relevant_chunks') as mock_retrieve:
            mock_retrieve.return_value = sample_chunks
            
            with patch.object(rag_service, 'generate_response') as mock_generate:
                mock_token_usage = TokenUsage(
                    prompt_tokens=50,
                    completion_tokens=30,
                    total_tokens=80
                )
                mock_generate.return_value = ("Test answer.", mock_token_usage)
                
                response = await rag_service.process_query(
                    query="Test query",
                    include_sources=False
                )
                
                assert len(response.source_chunks) == 0
    
    @pytest.mark.asyncio
    async def test_retrieve_relevant_chunks_success(self, rag_service, sample_chunks):
        """Test successful chunk retrieval."""
        # Mock embedding service
        rag_service.embedding_service.generate_query_embedding.return_value = [0.1] * 768
        
        # Mock vector storage search results (L2 distances)
        search_results = [
            (sample_chunks[0], 0.3),  # L2 distance
            (sample_chunks[1], 0.5),
            (sample_chunks[2], 0.7)
        ]
        rag_service.vector_storage.search.return_value = search_results
        
        # Retrieve chunks
        chunks = await rag_service.retrieve_relevant_chunks(
            query="test query",
            k=3
        )
        
        # Verify results
        assert len(chunks) == 3
        # Check that similarity scores were added to metadata
        for chunk in chunks:
            assert "similarity_score" in chunk.metadata
            assert 0.0 <= chunk.metadata["similarity_score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_retrieve_relevant_chunks_with_threshold(self, rag_service, sample_chunks):
        """Test chunk retrieval with similarity threshold filtering."""
        rag_service.similarity_threshold = 0.8  # High threshold
        
        # Mock embedding service
        rag_service.embedding_service.generate_query_embedding.return_value = [0.1] * 768
        
        # Mock search results with varying similarities
        search_results = [
            (sample_chunks[0], 0.2),  # High similarity (low L2 distance)
            (sample_chunks[1], 0.8),  # Medium similarity
            (sample_chunks[2], 1.2)   # Low similarity (high L2 distance)
        ]
        rag_service.vector_storage.search.return_value = search_results
        
        chunks = await rag_service.retrieve_relevant_chunks("test query", k=3)
        
        # Should only return chunks above threshold
        assert len(chunks) <= 2  # Only high and medium similarity chunks
    
    @pytest.mark.asyncio
    async def test_retrieve_relevant_chunks_embedding_error(self, rag_service):
        """Test chunk retrieval with embedding generation error."""
        rag_service.embedding_service.generate_query_embedding.side_effect = DocumentChunkingError("Embedding failed")
        
        with pytest.raises(RAGServiceError, match="Failed to generate query embedding"):
            await rag_service.retrieve_relevant_chunks("test query")
    
    @pytest.mark.asyncio
    async def test_retrieve_relevant_chunks_vector_search_error(self, rag_service):
        """Test chunk retrieval with vector search error."""
        rag_service.embedding_service.generate_query_embedding.return_value = [0.1] * 768
        rag_service.vector_storage.search.side_effect = VectorStorageError("Search failed")
        
        with pytest.raises(RAGServiceError, match="Failed to search vector storage"):
            await rag_service.retrieve_relevant_chunks("test query")
    
    def test_filter_and_rank_chunks(self, rag_service, sample_chunks):
        """Test chunk filtering and ranking."""
        # Add similarity scores to chunks
        for i, chunk in enumerate(sample_chunks):
            chunk.metadata["similarity_score"] = 0.9 - (i * 0.1)
        
        filtered_chunks = rag_service._filter_and_rank_chunks(sample_chunks, limit=2)
        
        # Should return top 2 chunks by similarity
        assert len(filtered_chunks) == 2
        assert filtered_chunks[0].metadata["similarity_score"] >= filtered_chunks[1].metadata["similarity_score"]
    
    def test_filter_and_rank_chunks_diversity(self, rag_service, sample_chunks):
        """Test chunk filtering with diversity consideration."""
        # All chunks have same high similarity but different documents
        for chunk in sample_chunks:
            chunk.metadata["similarity_score"] = 0.9
        
        filtered_chunks = rag_service._filter_and_rank_chunks(sample_chunks, limit=3)
        
        # Should prefer diversity across documents
        document_ids = [chunk.document_id for chunk in filtered_chunks]
        unique_docs = len(set(document_ids))
        assert unique_docs >= 2  # Should have chunks from multiple documents
    
    @pytest.mark.asyncio
    async def test_generate_response_success(self, rag_service, sample_chunks):
        """Test successful response generation."""
        mock_token_usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        
        with patch.object(rag_service.gemini_client, 'generate_response') as mock_generate:
            mock_generate.return_value = ("Generated response", mock_token_usage)
            
            response_text, token_usage = await rag_service.generate_response(
                query="Test query",
                context_chunks=sample_chunks
            )
            
            assert response_text == "Generated response"
            assert token_usage.total_tokens == 150
            
            # Verify context was built and passed
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args
            assert call_args[1]["context"] is not None
            assert "artificial intelligence" in call_args[1]["context"]
    
    @pytest.mark.asyncio
    async def test_generate_response_no_context(self, rag_service):
        """Test response generation with no context chunks."""
        mock_token_usage = TokenUsage(
            prompt_tokens=20,
            completion_tokens=10,
            total_tokens=30
        )
        
        with patch.object(rag_service.gemini_client, 'generate_response') as mock_generate:
            mock_generate.return_value = ("Response without context", mock_token_usage)
            
            response_text, token_usage = await rag_service.generate_response(
                query="Test query",
                context_chunks=[]
            )
            
            assert response_text == "Response without context"
            # Context should be empty string
            call_args = mock_generate.call_args
            assert call_args[1]["context"] == ""
    
    def test_build_context_string(self, rag_service, sample_chunks):
        """Test context string building."""
        context = rag_service._build_context_string(sample_chunks)
        
        assert "artificial intelligence" in context
        assert "machine learning" in context
        assert "deep learning" in context
        assert "[Context 1" in context
        assert "[Context 2" in context
        assert "ai_doc.pdf" in context
        assert "ml_doc.pdf" in context
    
    def test_build_context_string_empty(self, rag_service):
        """Test context string building with empty chunks."""
        context = rag_service._build_context_string([])
        assert context == ""
    
    def test_calculate_confidence_score(self, rag_service, sample_chunks):
        """Test confidence score calculation."""
        # Add similarity scores
        sample_chunks[0].metadata["similarity_score"] = 0.9
        sample_chunks[1].metadata["similarity_score"] = 0.8
        sample_chunks[2].metadata["similarity_score"] = 0.7
        
        confidence = rag_service._calculate_confidence_score(sample_chunks, "test query")
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably high with good similarity scores
    
    def test_calculate_confidence_score_no_chunks(self, rag_service):
        """Test confidence score calculation with no chunks."""
        confidence = rag_service._calculate_confidence_score([], "test query")
        assert confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self, rag_service):
        """Test health check with all components healthy."""
        # Mock successful health checks
        rag_service.vector_storage.get_stats.return_value = {
            "total_vectors": 100,
            "index_type": "HNSW"
        }
        rag_service.embedding_service.generate_query_embedding.return_value = [0.1] * 768
        rag_service.embedding_service.model_name = "test-model"
        
        with patch.object(rag_service.gemini_client, 'generate_response') as mock_generate:
            mock_token_usage = TokenUsage(prompt_tokens=5, completion_tokens=1, total_tokens=6)
            mock_generate.return_value = ("OK", mock_token_usage)
            
            health_status = await rag_service.health_check()
            
            assert health_status["status"] == "healthy"
            assert "vector_storage" in health_status["components"]
            assert "embedding_service" in health_status["components"]
            assert "gemini_api" in health_status["components"]
            
            for component in health_status["components"].values():
                assert component["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_health_check_degraded(self, rag_service):
        """Test health check with some components unhealthy."""
        # Mock vector storage failure
        rag_service.vector_storage.get_stats.side_effect = Exception("Storage error")
        
        # Mock successful embedding service
        rag_service.embedding_service.generate_query_embedding.return_value = [0.1] * 768
        rag_service.embedding_service.model_name = "test-model"
        
        # Mock successful Gemini API
        with patch.object(rag_service.gemini_client, 'generate_response') as mock_generate:
            mock_token_usage = TokenUsage(prompt_tokens=5, completion_tokens=1, total_tokens=6)
            mock_generate.return_value = ("OK", mock_token_usage)
            
            health_status = await rag_service.health_check()
            
            assert health_status["status"] == "degraded"
            assert health_status["components"]["vector_storage"]["status"] == "unhealthy"
            assert health_status["components"]["embedding_service"]["status"] == "healthy"
            assert health_status["components"]["gemini_api"]["status"] == "healthy"


class TestCreateRAGService:
    """Test cases for RAG service factory function."""
    
    @patch('src.services.rag_service.FAISSVectorStorage')
    @patch('src.services.rag_service.EmbeddingService')
    @patch('src.services.rag_service.settings')
    def test_create_rag_service_success(self, mock_settings, mock_embedding_cls, mock_vector_cls):
        """Test successful RAG service creation."""
        # Mock settings
        mock_settings.vector_store.embedding_dimension = 768
        mock_settings.embedding.batch_size = 32
        mock_settings.llm.api_key = "test-api-key"
        
        # Mock service instances
        mock_vector_storage = Mock()
        mock_embedding_service = Mock()
        mock_vector_cls.return_value = mock_vector_storage
        mock_embedding_cls.return_value = mock_embedding_service
        
        # Create service
        service = create_rag_service(
            vector_storage_path="test/path",
            embedding_model="test-model",
            gemini_model="gemini-pro"
        )
        
        # Verify creation
        assert isinstance(service, RAGService)
        mock_vector_cls.assert_called_once_with(
            dimension=768,
            storage_path="test/path"
        )
        mock_embedding_cls.assert_called_once_with(
            model_name="test-model",
            batch_size=32
        )
    
    @patch('src.services.rag_service.FAISSVectorStorage')
    @patch('src.services.rag_service.EmbeddingService')
    @patch('src.services.rag_service.settings')
    def test_create_rag_service_no_api_key(self, mock_settings, mock_embedding_cls, mock_vector_cls):
        """Test RAG service creation without API key."""
        # Mock settings without API key
        mock_settings.vector_store.embedding_dimension = 768
        mock_settings.embedding.batch_size = 32
        mock_settings.llm.api_key = None
        
        with pytest.raises(RAGServiceError, match="Gemini API key not provided"):
            create_rag_service()
    
    @patch('src.services.rag_service.FAISSVectorStorage')
    @patch('src.services.rag_service.settings')
    def test_create_rag_service_vector_storage_error(self, mock_settings, mock_vector_cls):
        """Test RAG service creation with vector storage initialization error."""
        mock_settings.vector_store.embedding_dimension = 768
        mock_settings.llm.api_key = "test-api-key"
        
        # Mock vector storage initialization failure
        mock_vector_cls.side_effect = Exception("Vector storage init failed")
        
        with pytest.raises(RAGServiceError, match="Failed to create RAG service"):
            create_rag_service()


# Integration test fixtures and helpers
@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.mark.asyncio
async def test_rag_service_integration_flow():
    """Integration test for complete RAG service flow."""
    # This test would require actual service instances
    # For now, it's a placeholder for future integration testing
    pass


if __name__ == "__main__":
    pytest.main([__file__])