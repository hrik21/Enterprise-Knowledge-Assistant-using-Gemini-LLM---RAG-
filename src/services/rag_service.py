"""
RAG Service Core Logic for the RAG Enterprise Assistant.

This module provides the core RAG (Retrieval-Augmented Generation) functionality
that combines FAISS vector search with Gemini API for intelligent question answering.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
import json
import httpx
from datetime import datetime

from ..models.data_models import DocumentChunk, RAGResponse, TokenUsage, QueryRequest
from ..services.vector_storage import FAISSVectorStorage, VectorStorageError
from ..services.document_chunking import EmbeddingService, DocumentChunkingError
from config.settings import settings

logger = logging.getLogger(__name__)


class RAGServiceError(Exception):
    """Custom exception for RAG service operations."""
    pass


class GeminiAPIClient:
    """
    Client for interacting with Google Gemini API.
    
    Handles authentication, request formatting, and response parsing
    for the Gemini language model API.
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        """
        Initialize Gemini API client.
        
        Args:
            api_key: Google API key for Gemini
            model_name: Name of the Gemini model to use
        """
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        
        # HTTP client with timeout and retry configuration
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        
        logger.info(f"Initialized GeminiAPIClient with model {model_name}")
    
    async def generate_response(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        context: Optional[str] = None
    ) -> Tuple[str, TokenUsage]:
        """
        Generate response using Gemini API.
        
        Args:
            prompt: User query/prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            context: Optional context information
            
        Returns:
            Tuple of (response_text, token_usage)
            
        Raises:
            RAGServiceError: If API call fails
        """
        try:
            # Construct the full prompt with context
            full_prompt = self._construct_prompt(prompt, context)
            
            # Prepare API request
            url = f"{self.base_url}/models/{self.model_name}:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key
            }
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": full_prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                    "topP": 0.8,
                    "topK": 10
                }
            }
            
            # Make API request with retry logic
            response = await self._make_request_with_retry(url, headers, payload)
            
            # Parse response
            response_text, token_usage = self._parse_response(response, full_prompt)
            
            logger.info(f"Generated response with {token_usage.total_tokens} tokens")
            return response_text, token_usage
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise RAGServiceError(f"Failed to generate response: {e}")
    
    def _construct_prompt(self, query: str, context: Optional[str] = None) -> str:
        """
        Construct the full prompt with context and instructions.
        
        Args:
            query: User query
            context: Retrieved context information
            
        Returns:
            str: Formatted prompt
        """
        if context:
            prompt = f"""You are an intelligent assistant that answers questions based on provided context. 
Use the following context to answer the user's question. If the context doesn't contain enough information 
to answer the question, say so clearly.

Context:
{context}

Question: {query}

Answer:"""
        else:
            prompt = f"""You are an intelligent assistant. Please answer the following question:

Question: {query}

Answer:"""
        
        return prompt
    
    async def _make_request_with_retry(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        max_retries: int = 3
    ) -> httpx.Response:
        """
        Make HTTP request with exponential backoff retry.
        
        Args:
            url: API endpoint URL
            headers: Request headers
            payload: Request payload
            max_retries: Maximum number of retries
            
        Returns:
            httpx.Response: API response
            
        Raises:
            RAGServiceError: If all retries fail
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                response = await self.client.post(url, headers=headers, json=payload)
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:  # Rate limit
                    if attempt < max_retries:
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}")
                        await asyncio.sleep(wait_time)
                        continue
                else:
                    response.raise_for_status()
                    
            except httpx.HTTPError as e:
                last_exception = e
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(f"Request failed, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    break
        
        raise RAGServiceError(f"API request failed after {max_retries} retries: {last_exception}")
    
    def _parse_response(self, response: httpx.Response, prompt: str) -> Tuple[str, TokenUsage]:
        """
        Parse Gemini API response and extract text and token usage.
        
        Args:
            response: HTTP response from Gemini API
            prompt: Original prompt for token counting
            
        Returns:
            Tuple of (response_text, token_usage)
        """
        try:
            data = response.json()
            
            # Extract generated text
            candidates = data.get("candidates", [])
            if not candidates:
                raise RAGServiceError("No candidates in API response")
            
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                raise RAGServiceError("No parts in API response")
            
            response_text = parts[0].get("text", "").strip()
            if not response_text:
                raise RAGServiceError("Empty response from API")
            
            # Extract token usage (Gemini API may not always provide this)
            usage_metadata = data.get("usageMetadata", {})
            prompt_tokens = usage_metadata.get("promptTokenCount", self._estimate_tokens(prompt))
            completion_tokens = usage_metadata.get("candidatesTokenCount", self._estimate_tokens(response_text))
            total_tokens = usage_metadata.get("totalTokenCount", prompt_tokens + completion_tokens)
            
            token_usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
            
            return response_text, token_usage
            
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse API response: {e}")
            raise RAGServiceError(f"Invalid API response format: {e}")
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text (rough approximation).
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            int: Estimated token count
        """
        # Rough estimation: 1 token ≈ 0.75 words
        word_count = len(text.split())
        return max(1, int(word_count * 1.33))
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class RAGService:
    """
    Core RAG service that orchestrates document retrieval and response generation.
    
    Combines FAISS vector search with metadata filtering and Gemini API
    for intelligent question answering based on document corpus.
    """
    
    def __init__(
        self,
        vector_storage: FAISSVectorStorage,
        embedding_service: EmbeddingService,
        gemini_api_key: str,
        gemini_model: str = "gemini-pro",
        max_context_chunks: int = 5,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize RAG service with required components.
        
        Args:
            vector_storage: FAISS vector storage instance
            embedding_service: Embedding generation service
            gemini_api_key: API key for Gemini
            gemini_model: Gemini model name
            max_context_chunks: Maximum chunks to use for context
            similarity_threshold: Minimum similarity score for retrieval
        """
        self.vector_storage = vector_storage
        self.embedding_service = embedding_service
        self.max_context_chunks = max_context_chunks
        self.similarity_threshold = similarity_threshold
        
        # Initialize Gemini API client
        self.gemini_client = GeminiAPIClient(gemini_api_key, gemini_model)
        
        logger.info(f"Initialized RAGService with {gemini_model}, max_chunks={max_context_chunks}")
    
    async def process_query(
        self,
        query: str,
        context_limit: int = 5,
        include_sources: bool = True,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> RAGResponse:
        """
        Process a user query through the complete RAG pipeline.
        
        Args:
            query: User's natural language question
            context_limit: Maximum number of context chunks to retrieve
            include_sources: Whether to include source chunks in response
            metadata_filters: Optional filters for document retrieval
            
        Returns:
            RAGResponse: Complete response with answer and metadata
            
        Raises:
            RAGServiceError: If query processing fails
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not query or not query.strip():
                raise RAGServiceError("Query cannot be empty")
            
            query = query.strip()
            context_limit = min(context_limit, self.max_context_chunks)
            
            logger.info(f"Processing query: '{query[:100]}...' with context_limit={context_limit}")
            
            # Step 1: Retrieve relevant document chunks
            relevant_chunks = await self.retrieve_relevant_chunks(
                query, 
                k=context_limit * 2,  # Retrieve more for filtering
                metadata_filters=metadata_filters
            )
            
            # Step 2: Filter and rank chunks
            filtered_chunks = self._filter_and_rank_chunks(relevant_chunks, context_limit)
            
            # Step 3: Generate response using retrieved context
            answer, token_usage = await self.generate_response(query, filtered_chunks)
            
            # Step 4: Calculate confidence score
            confidence_score = self._calculate_confidence_score(filtered_chunks, query)
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Create response
            response = RAGResponse(
                query=query,
                answer=answer,
                source_chunks=filtered_chunks if include_sources else [],
                confidence_score=confidence_score,
                processing_time_ms=processing_time_ms,
                token_usage=token_usage
            )
            
            logger.info(f"Successfully processed query in {processing_time_ms}ms")
            return response
            
        except Exception as e:
            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Query processing failed after {processing_time_ms}ms: {e}")
            raise RAGServiceError(f"Failed to process query: {e}")
    
    async def retrieve_relevant_chunks(
        self,
        query: str,
        k: int = 10,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Retrieve relevant document chunks using vector similarity search.
        
        Args:
            query: Search query
            k: Number of chunks to retrieve
            metadata_filters: Optional metadata filters
            
        Returns:
            List[DocumentChunk]: Retrieved document chunks
            
        Raises:
            RAGServiceError: If retrieval fails
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.generate_query_embedding(query)
            
            # Search vector storage
            search_results = self.vector_storage.search(
                query_vector=query_embedding,
                k=k,
                filter_metadata=metadata_filters
            )
            
            # Extract chunks and apply similarity threshold
            relevant_chunks = []
            for chunk, similarity_score in search_results:
                # Convert L2 distance to similarity score (for normalized vectors)
                # L2 distance of normalized vectors: d = sqrt(2 * (1 - cosine_similarity))
                # So: cosine_similarity = 1 - (d^2 / 2)
                cosine_similarity = max(0.0, 1.0 - (similarity_score ** 2 / 2))
                
                if cosine_similarity >= self.similarity_threshold:
                    # Add similarity score to chunk metadata
                    chunk_with_score = DocumentChunk(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        content=chunk.content,
                        metadata={**chunk.metadata, "similarity_score": cosine_similarity},
                        embedding=chunk.embedding,
                        created_at=chunk.created_at,
                        chunk_index=chunk.chunk_index,
                        token_count=chunk.token_count
                    )
                    relevant_chunks.append(chunk_with_score)
            
            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks (threshold: {self.similarity_threshold})")
            return relevant_chunks
            
        except DocumentChunkingError as e:
            logger.error(f"Embedding generation failed: {e}")
            raise RAGServiceError(f"Failed to generate query embedding: {e}")
        except VectorStorageError as e:
            logger.error(f"Vector search failed: {e}")
            raise RAGServiceError(f"Failed to search vector storage: {e}")
        except Exception as e:
            logger.error(f"Chunk retrieval failed: {e}")
            raise RAGServiceError(f"Failed to retrieve chunks: {e}")
    
    def _filter_and_rank_chunks(
        self,
        chunks: List[DocumentChunk],
        limit: int
    ) -> List[DocumentChunk]:
        """
        Filter and rank retrieved chunks for optimal context.
        
        Args:
            chunks: Retrieved document chunks
            limit: Maximum number of chunks to return
            
        Returns:
            List[DocumentChunk]: Filtered and ranked chunks
        """
        if not chunks:
            return []
        
        # Sort by similarity score (descending)
        sorted_chunks = sorted(
            chunks,
            key=lambda x: x.metadata.get("similarity_score", 0.0),
            reverse=True
        )
        
        # Apply diversity filtering to avoid too many chunks from same document
        diverse_chunks = []
        document_counts = {}
        max_per_document = max(1, limit // 2)  # Allow at most half from same document
        
        for chunk in sorted_chunks:
            doc_id = chunk.document_id
            doc_count = document_counts.get(doc_id, 0)
            
            if doc_count < max_per_document and len(diverse_chunks) < limit:
                diverse_chunks.append(chunk)
                document_counts[doc_id] = doc_count + 1
        
        # If we still need more chunks and have space, add remaining high-scoring ones
        if len(diverse_chunks) < limit:
            for chunk in sorted_chunks:
                if chunk not in diverse_chunks and len(diverse_chunks) < limit:
                    diverse_chunks.append(chunk)
        
        logger.info(f"Filtered to {len(diverse_chunks)} diverse chunks from {len(set(c.document_id for c in diverse_chunks))} documents")
        return diverse_chunks[:limit]
    
    async def generate_response(
        self,
        query: str,
        context_chunks: List[DocumentChunk]
    ) -> Tuple[str, TokenUsage]:
        """
        Generate response using Gemini API with retrieved context.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            
        Returns:
            Tuple of (response_text, token_usage)
            
        Raises:
            RAGServiceError: If response generation fails
        """
        try:
            # Construct context from chunks
            context = self._build_context_string(context_chunks)
            
            # Generate response using Gemini API
            response_text, token_usage = await self.gemini_client.generate_response(
                prompt=query,
                max_tokens=settings.llm.max_tokens,
                temperature=settings.llm.temperature,
                context=context
            )
            
            return response_text, token_usage
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            raise RAGServiceError(f"Failed to generate response: {e}")
    
    def _build_context_string(self, chunks: List[DocumentChunk]) -> str:
        """
        Build context string from document chunks.
        
        Args:
            chunks: Document chunks to use as context
            
        Returns:
            str: Formatted context string
        """
        if not chunks:
            return ""
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            # Include document metadata for better context
            doc_info = ""
            if "file_name" in chunk.metadata:
                doc_info = f" (from {chunk.metadata['file_name']})"
            
            similarity_score = chunk.metadata.get("similarity_score", 0.0)
            
            context_part = f"[Context {i}{doc_info}, relevance: {similarity_score:.2f}]\n{chunk.content}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _calculate_confidence_score(
        self,
        chunks: List[DocumentChunk],
        query: str
    ) -> float:
        """
        Calculate confidence score for the response based on retrieval quality.
        
        Args:
            chunks: Retrieved chunks used for response
            query: Original query
            
        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        if not chunks:
            return 0.0
        
        # Base confidence on average similarity score
        similarity_scores = [
            chunk.metadata.get("similarity_score", 0.0) 
            for chunk in chunks
        ]
        
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        
        # Adjust based on number of chunks (more context = higher confidence)
        chunk_factor = min(1.0, len(chunks) / self.max_context_chunks)
        
        # Adjust based on diversity of sources
        unique_docs = len(set(chunk.document_id for chunk in chunks))
        diversity_factor = min(1.0, unique_docs / max(1, len(chunks) // 2))
        
        # Combine factors
        confidence = avg_similarity * 0.7 + chunk_factor * 0.2 + diversity_factor * 0.1
        
        return min(1.0, max(0.0, confidence))
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on RAG service components.
        
        Returns:
            Dict containing health status of all components
        """
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        try:
            # Check vector storage
            vector_stats = self.vector_storage.get_stats()
            health_status["components"]["vector_storage"] = {
                "status": "healthy",
                "total_vectors": vector_stats["total_vectors"],
                "index_type": vector_stats["index_type"]
            }
        except Exception as e:
            health_status["components"]["vector_storage"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        try:
            # Check embedding service
            test_embedding = self.embedding_service.generate_query_embedding("test")
            health_status["components"]["embedding_service"] = {
                "status": "healthy",
                "model": self.embedding_service.model_name,
                "dimension": len(test_embedding)
            }
        except Exception as e:
            health_status["components"]["embedding_service"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        try:
            # Check Gemini API (simple test)
            test_response, _ = await self.gemini_client.generate_response(
                "Say 'OK' if you can respond.",
                max_tokens=10,
                temperature=0.0
            )
            health_status["components"]["gemini_api"] = {
                "status": "healthy",
                "model": self.gemini_client.model_name,
                "test_response_length": len(test_response)
            }
        except Exception as e:
            health_status["components"]["gemini_api"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # Overall status
        unhealthy_components = [
            comp for comp, status in health_status["components"].items()
            if status["status"] == "unhealthy"
        ]
        
        if unhealthy_components:
            if len(unhealthy_components) == len(health_status["components"]):
                health_status["status"] = "unhealthy"
            else:
                health_status["status"] = "degraded"
        
        return health_status
    
    async def close(self):
        """Clean up resources."""
        await self.gemini_client.close()
        logger.info("RAG service closed")


def create_rag_service(
    vector_storage_path: str = "data/vector_storage",
    embedding_model: str = "all-MiniLM-L6-v2",
    gemini_api_key: Optional[str] = None,
    gemini_model: str = "gemini-pro"
) -> RAGService:
    """
    Factory function to create a configured RAG service instance.
    
    Args:
        vector_storage_path: Path to vector storage
        embedding_model: Sentence-transformers model name
        gemini_api_key: Gemini API key (uses settings if not provided)
        gemini_model: Gemini model name
        
    Returns:
        RAGService: Configured RAG service instance
        
    Raises:
        RAGServiceError: If initialization fails
    """
    try:
        # Initialize vector storage
        vector_storage = FAISSVectorStorage(
            dimension=settings.vector_store.embedding_dimension,
            storage_path=vector_storage_path
        )
        
        # Initialize embedding service
        embedding_service = EmbeddingService(
            model_name=embedding_model,
            batch_size=settings.embedding.batch_size
        )
        
        # Use provided API key or get from settings
        api_key = gemini_api_key or settings.llm.api_key
        if not api_key:
            raise RAGServiceError("Gemini API key not provided")
        
        # Create RAG service
        rag_service = RAGService(
            vector_storage=vector_storage,
            embedding_service=embedding_service,
            gemini_api_key=api_key,
            gemini_model=gemini_model,
            max_context_chunks=5,
            similarity_threshold=0.7
        )
        
        logger.info("Successfully created RAG service")
        return rag_service
        
    except Exception as e:
        logger.error(f"Failed to create RAG service: {e}")
        raise RAGServiceError(f"Failed to create RAG service: {e}")