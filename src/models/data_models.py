"""
Core data models for the RAG Enterprise Assistant system.

This module contains Pydantic models for data validation and serialization
across the entire system, including document chunks, RAG responses, and metadata.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, ConfigDict


class ProcessingStatus(str, Enum):
    """Status enumeration for document processing pipeline."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class TokenUsage(BaseModel):
    """Token usage statistics for API calls."""
    prompt_tokens: int = Field(..., ge=0, description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., ge=0, description="Number of tokens in the completion")
    total_tokens: int = Field(..., ge=0, description="Total number of tokens used")
    
    @field_validator('total_tokens')
    @classmethod
    def validate_total_tokens(cls, v, info):
        """Ensure total tokens equals sum of prompt and completion tokens."""
        if info.data and 'prompt_tokens' in info.data and 'completion_tokens' in info.data:
            expected_total = info.data['prompt_tokens'] + info.data['completion_tokens']
            if v != expected_total:
                raise ValueError(f"Total tokens ({v}) must equal prompt_tokens + completion_tokens ({expected_total})")
        return v


class DocumentChunk(BaseModel):
    """
    Represents a chunk of a document with its content and metadata.
    
    This model is used throughout the system for storing and retrieving
    document segments that have been processed and embedded.
    """
    chunk_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the chunk")
    document_id: str = Field(..., description="ID of the parent document")
    content: str = Field(..., min_length=1, max_length=10000, description="Text content of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the chunk")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the chunk content")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when chunk was created")
    chunk_index: int = Field(..., ge=0, description="Index of this chunk within the parent document")
    token_count: int = Field(..., ge=0, description="Number of tokens in the chunk content")
    
    @field_validator('embedding')
    @classmethod
    def validate_embedding_dimension(cls, v):
        """Ensure embedding has correct dimensions if provided."""
        if v is not None:
            # Allow common embedding dimensions (384, 512, 768, 1024, 1536)
            valid_dimensions = [384, 512, 768, 1024, 1536]
            if len(v) not in valid_dimensions:
                raise ValueError(f"Embedding dimension {len(v)} not in supported dimensions: {valid_dimensions}")
        return v
    
    @field_validator('content')
    @classmethod
    def validate_content_not_empty(cls, v):
        """Ensure content is not just whitespace."""
        if not v.strip():
            raise ValueError("Content cannot be empty or only whitespace")
        return v.strip()
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


class DocumentMetadata(BaseModel):
    """
    Metadata information for documents in the system.
    
    Tracks document processing status, file information, and organizational tags.
    """
    document_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the document")
    filename: str = Field(..., min_length=1, description="Original filename of the document")
    file_type: str = Field(..., description="MIME type or file extension")
    file_size: int = Field(..., ge=0, description="Size of the file in bytes")
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the document was uploaded")
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="Current processing status")
    chunk_count: int = Field(default=0, ge=0, description="Number of chunks created from this document")
    tags: List[str] = Field(default_factory=list, description="Organizational tags for the document")
    
    @field_validator('file_type')
    @classmethod
    def validate_file_type(cls, v):
        """Ensure file type is in supported formats."""
        supported_types = {'.pdf', '.txt', '.docx', '.doc', '.md', 'application/pdf', 'text/plain', 'text/markdown'}
        if v.lower() not in supported_types:
            raise ValueError(f"Unsupported file type: {v}. Supported types: {supported_types}")
        return v.lower()
    
    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v):
        """Ensure tags are non-empty strings and normalize them."""
        if v:
            normalized_tags = []
            for tag in v:
                if not isinstance(tag, str):
                    raise ValueError("Tags must be non-empty strings")
                stripped_tag = tag.strip()
                if stripped_tag:  # Only add non-empty tags
                    normalized_tags.append(stripped_tag.lower())
            return normalized_tags
        return v
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


class RAGResponse(BaseModel):
    """
    Response model for RAG queries containing the generated answer and metadata.
    
    Includes the original query, generated response, source documents,
    and performance metrics.
    """
    query: str = Field(..., min_length=1, description="Original user query")
    answer: str = Field(..., min_length=1, description="Generated response from the RAG system")
    source_chunks: List[DocumentChunk] = Field(default_factory=list, description="Document chunks used to generate the response")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the response")
    processing_time_ms: int = Field(..., ge=0, description="Time taken to process the query in milliseconds")
    token_usage: TokenUsage = Field(..., description="Token usage statistics for the query")
    
    @field_validator('query')
    @classmethod
    def validate_query_not_empty(cls, v):
        """Ensure query is not just whitespace."""
        if not v.strip():
            raise ValueError("Query cannot be empty or only whitespace")
        return v.strip()
    
    @field_validator('answer')
    @classmethod
    def validate_answer_not_empty(cls, v):
        """Ensure answer is not just whitespace."""
        if not v.strip():
            raise ValueError("Answer cannot be empty or only whitespace")
        return v.strip()
    
    @field_validator('source_chunks')
    @classmethod
    def validate_source_chunks_limit(cls, v):
        """Ensure reasonable number of source chunks."""
        if len(v) > 20:  # Reasonable limit for context
            raise ValueError("Too many source chunks (max 20)")
        return v
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    query: str = Field(..., min_length=1, max_length=1000, description="User query text")
    context_limit: int = Field(default=5, ge=1, le=20, description="Maximum number of context chunks to retrieve")
    include_sources: bool = Field(default=True, description="Whether to include source chunks in response")
    
    @field_validator('query')
    @classmethod
    def validate_query_not_empty(cls, v):
        """Ensure query is not just whitespace."""
        if not v.strip():
            raise ValueError("Query cannot be empty or only whitespace")
        return v.strip()


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    version: str = Field(..., description="Application version")
    dependencies: Dict[str, str] = Field(default_factory=dict, description="Status of external dependencies")
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )