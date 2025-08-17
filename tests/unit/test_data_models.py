"""
Unit tests for core data models.

Tests validation logic, serialization, and edge cases for all Pydantic models
used throughout the RAG Enterprise Assistant system.
"""

import json
import pytest
from datetime import datetime
from typing import Dict, Any
from uuid import uuid4

from pydantic import ValidationError

from src.models.data_models import (
    DocumentChunk,
    DocumentMetadata,
    RAGResponse,
    QueryRequest,
    HealthCheckResponse,
    TokenUsage,
    ProcessingStatus
)


class TestTokenUsage:
    """Test cases for TokenUsage model."""
    
    def test_valid_token_usage(self):
        """Test creating valid TokenUsage instance."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
    
    def test_invalid_total_tokens(self):
        """Test validation error for incorrect total tokens."""
        with pytest.raises(ValidationError) as exc_info:
            TokenUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=200  # Should be 150
            )
        assert "Total tokens" in str(exc_info.value)
    
    def test_negative_tokens(self):
        """Test validation error for negative token counts."""
        with pytest.raises(ValidationError):
            TokenUsage(
                prompt_tokens=-10,
                completion_tokens=50,
                total_tokens=40
            )


class TestDocumentChunk:
    """Test cases for DocumentChunk model."""
    
    def test_valid_document_chunk(self):
        """Test creating valid DocumentChunk instance."""
        chunk = DocumentChunk(
            document_id="doc123",
            content="This is a test chunk content.",
            chunk_index=0,
            token_count=6
        )
        assert chunk.document_id == "doc123"
        assert chunk.content == "This is a test chunk content."
        assert chunk.chunk_index == 0
        assert chunk.token_count == 6
        assert chunk.chunk_id is not None
        assert isinstance(chunk.created_at, datetime)
    
    def test_empty_content_validation(self):
        """Test validation error for empty content."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                document_id="doc123",
                content="   ",  # Only whitespace
                chunk_index=0,
                token_count=0
            )
        assert "Content cannot be empty" in str(exc_info.value)
    
    def test_content_too_long(self):
        """Test validation error for content exceeding max length."""
        long_content = "x" * 10001  # Exceeds max_length of 10000
        with pytest.raises(ValidationError):
            DocumentChunk(
                document_id="doc123",
                content=long_content,
                chunk_index=0,
                token_count=10001
            )
    
    def test_negative_chunk_index(self):
        """Test validation error for negative chunk index."""
        with pytest.raises(ValidationError):
            DocumentChunk(
                document_id="doc123",
                content="Valid content",
                chunk_index=-1,
                token_count=2
            )
    
    def test_valid_embedding(self):
        """Test valid embedding with correct dimensions."""
        embedding = [0.1] * 768  # Correct dimension
        chunk = DocumentChunk(
            document_id="doc123",
            content="Test content",
            chunk_index=0,
            token_count=2,
            embedding=embedding
        )
        assert len(chunk.embedding) == 768
    
    def test_invalid_embedding_dimension(self):
        """Test validation error for incorrect embedding dimensions."""
        embedding = [0.1] * 512  # Wrong dimension
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                document_id="doc123",
                content="Test content",
                chunk_index=0,
                token_count=2,
                embedding=embedding
            )
        assert "768 dimensions" in str(exc_info.value)
    
    def test_invalid_embedding_values(self):
        """Test validation error for non-numeric embedding values."""
        embedding = ["invalid"] * 768
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(
                document_id="doc123",
                content="Test content",
                chunk_index=0,
                token_count=2,
                embedding=embedding
            )
        assert "valid number" in str(exc_info.value)
    
    def test_serialization(self):
        """Test JSON serialization of DocumentChunk."""
        chunk = DocumentChunk(
            document_id="doc123",
            content="Test content",
            chunk_index=0,
            token_count=2
        )
        json_str = chunk.model_dump_json()
        data = json.loads(json_str)
        
        assert data["document_id"] == "doc123"
        assert data["content"] == "Test content"
        assert "created_at" in data


class TestDocumentMetadata:
    """Test cases for DocumentMetadata model."""
    
    def test_valid_document_metadata(self):
        """Test creating valid DocumentMetadata instance."""
        metadata = DocumentMetadata(
            filename="test.pdf",
            file_type=".pdf",
            file_size=1024,
            tags=["important", "research"]
        )
        assert metadata.filename == "test.pdf"
        assert metadata.file_type == ".pdf"
        assert metadata.file_size == 1024
        assert metadata.tags == ["important", "research"]
        assert metadata.processing_status == ProcessingStatus.PENDING
    
    def test_unsupported_file_type(self):
        """Test validation error for unsupported file type."""
        with pytest.raises(ValidationError) as exc_info:
            DocumentMetadata(
                filename="test.xyz",
                file_type=".xyz",
                file_size=1024
            )
        assert "Unsupported file type" in str(exc_info.value)
    
    def test_negative_file_size(self):
        """Test validation error for negative file size."""
        with pytest.raises(ValidationError):
            DocumentMetadata(
                filename="test.pdf",
                file_type=".pdf",
                file_size=-100
            )
    
    def test_empty_filename(self):
        """Test validation error for empty filename."""
        with pytest.raises(ValidationError):
            DocumentMetadata(
                filename="",
                file_type=".pdf",
                file_size=1024
            )
    
    def test_tag_validation(self):
        """Test tag validation and normalization."""
        metadata = DocumentMetadata(
            filename="test.pdf",
            file_type=".pdf",
            file_size=1024,
            tags=["  Important  ", "RESEARCH", "", "   "]
        )
        # Should normalize to lowercase and remove empty tags
        assert metadata.tags == ["important", "research"]
    
    def test_invalid_tags(self):
        """Test validation error for invalid tag types."""
        with pytest.raises(ValidationError):
            DocumentMetadata(
                filename="test.pdf",
                file_type=".pdf",
                file_size=1024,
                tags=[123, "valid_tag"]  # Non-string tag
            )


class TestRAGResponse:
    """Test cases for RAGResponse model."""
    
    def test_valid_rag_response(self):
        """Test creating valid RAGResponse instance."""
        chunk = DocumentChunk(
            document_id="doc123",
            content="Source content",
            chunk_index=0,
            token_count=2
        )
        token_usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        
        response = RAGResponse(
            query="What is the answer?",
            answer="The answer is 42.",
            source_chunks=[chunk],
            confidence_score=0.95,
            processing_time_ms=250,
            token_usage=token_usage
        )
        
        assert response.query == "What is the answer?"
        assert response.answer == "The answer is 42."
        assert len(response.source_chunks) == 1
        assert response.confidence_score == 0.95
        assert response.processing_time_ms == 250
    
    def test_empty_query_validation(self):
        """Test validation error for empty query."""
        token_usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        
        with pytest.raises(ValidationError) as exc_info:
            RAGResponse(
                query="   ",  # Only whitespace
                answer="Valid answer",
                confidence_score=0.8,
                processing_time_ms=100,
                token_usage=token_usage
            )
        assert "Query cannot be empty" in str(exc_info.value)
    
    def test_empty_answer_validation(self):
        """Test validation error for empty answer."""
        token_usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        
        with pytest.raises(ValidationError) as exc_info:
            RAGResponse(
                query="Valid query",
                answer="   ",  # Whitespace only answer
                confidence_score=0.8,
                processing_time_ms=100,
                token_usage=token_usage
            )
        assert "Answer cannot be empty" in str(exc_info.value)
    
    def test_confidence_score_bounds(self):
        """Test validation for confidence score bounds."""
        token_usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        
        # Test score > 1.0
        with pytest.raises(ValidationError):
            RAGResponse(
                query="Valid query",
                answer="Valid answer",
                confidence_score=1.5,
                processing_time_ms=100,
                token_usage=token_usage
            )
        
        # Test score < 0.0
        with pytest.raises(ValidationError):
            RAGResponse(
                query="Valid query",
                answer="Valid answer",
                confidence_score=-0.1,
                processing_time_ms=100,
                token_usage=token_usage
            )
    
    def test_too_many_source_chunks(self):
        """Test validation error for too many source chunks."""
        chunks = [
            DocumentChunk(
                document_id=f"doc{i}",
                content=f"Content {i}",
                chunk_index=0,
                token_count=2
            ) for i in range(21)  # Exceeds limit of 20
        ]
        token_usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        
        with pytest.raises(ValidationError) as exc_info:
            RAGResponse(
                query="Valid query",
                answer="Valid answer",
                source_chunks=chunks,
                confidence_score=0.8,
                processing_time_ms=100,
                token_usage=token_usage
            )
        assert "Too many source chunks" in str(exc_info.value)


class TestQueryRequest:
    """Test cases for QueryRequest model."""
    
    def test_valid_query_request(self):
        """Test creating valid QueryRequest instance."""
        request = QueryRequest(
            query="What is machine learning?",
            context_limit=10,
            include_sources=True
        )
        assert request.query == "What is machine learning?"
        assert request.context_limit == 10
        assert request.include_sources is True
    
    def test_default_values(self):
        """Test default values for QueryRequest."""
        request = QueryRequest(query="Test query")
        assert request.context_limit == 5
        assert request.include_sources is True
    
    def test_empty_query_validation(self):
        """Test validation error for empty query."""
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(query="   ")  # Only whitespace
        assert "Query cannot be empty" in str(exc_info.value)
    
    def test_query_too_long(self):
        """Test validation error for query exceeding max length."""
        long_query = "x" * 1001  # Exceeds max_length of 1000
        with pytest.raises(ValidationError):
            QueryRequest(query=long_query)
    
    def test_context_limit_bounds(self):
        """Test validation for context limit bounds."""
        # Test context_limit < 1
        with pytest.raises(ValidationError):
            QueryRequest(query="Valid query", context_limit=0)
        
        # Test context_limit > 20
        with pytest.raises(ValidationError):
            QueryRequest(query="Valid query", context_limit=21)


class TestHealthCheckResponse:
    """Test cases for HealthCheckResponse model."""
    
    def test_valid_health_check_response(self):
        """Test creating valid HealthCheckResponse instance."""
        response = HealthCheckResponse(
            status="healthy",
            version="1.0.0",
            dependencies={"database": "connected", "api": "available"}
        )
        assert response.status == "healthy"
        assert response.version == "1.0.0"
        assert response.dependencies["database"] == "connected"
        assert isinstance(response.timestamp, datetime)
    
    def test_serialization_with_datetime(self):
        """Test JSON serialization with datetime fields."""
        response = HealthCheckResponse(
            status="healthy",
            version="1.0.0"
        )
        json_str = response.model_dump_json()
        data = json.loads(json_str)
        
        assert "timestamp" in data
        assert isinstance(data["timestamp"], str)  # Should be ISO format string


class TestProcessingStatus:
    """Test cases for ProcessingStatus enum."""
    
    def test_processing_status_values(self):
        """Test all ProcessingStatus enum values."""
        assert ProcessingStatus.PENDING == "pending"
        assert ProcessingStatus.PROCESSING == "processing"
        assert ProcessingStatus.COMPLETED == "completed"
        assert ProcessingStatus.FAILED == "failed"
        assert ProcessingStatus.RETRYING == "retrying"
    
    def test_processing_status_in_metadata(self):
        """Test using ProcessingStatus in DocumentMetadata."""
        metadata = DocumentMetadata(
            filename="test.pdf",
            file_type=".pdf",
            file_size=1024,
            processing_status=ProcessingStatus.COMPLETED
        )
        assert metadata.processing_status == ProcessingStatus.COMPLETED