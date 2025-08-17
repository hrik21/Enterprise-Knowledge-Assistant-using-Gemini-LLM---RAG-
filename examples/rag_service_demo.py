"""
RAG Service Demo Script

Demonstrates the core functionality of the RAG service including:
- Document chunk processing
- Vector similarity search
- Response generation with Gemini API
- Error handling and health checks

This is a demonstration script that shows how to use the RAG service
with mocked dependencies for testing purposes.
"""

import asyncio
import logging
from typing import List
from unittest.mock import Mock, AsyncMock
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock heavy dependencies for demo
import sys
from unittest.mock import MagicMock

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
mock_settings.llm.api_key = "demo-api-key"
sys.modules['config'] = MagicMock()
sys.modules['config.settings'] = MagicMock()
sys.modules['config.settings'].settings = mock_settings

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.rag_service import RAGService, GeminiAPIClient, RAGServiceError
from src.models.data_models import DocumentChunk, RAGResponse, TokenUsage


def create_sample_chunks() -> List[DocumentChunk]:
    """Create sample document chunks for demonstration."""
    return [
        DocumentChunk(
            chunk_id="chunk_ai_1",
            document_id="ai_fundamentals",
            content="""
            Artificial Intelligence (AI) is a branch of computer science that aims to create 
            intelligent machines that can perform tasks that typically require human intelligence. 
            These tasks include learning, reasoning, problem-solving, perception, and language understanding.
            """.strip(),
            metadata={
                "file_name": "ai_fundamentals.pdf",
                "section": "introduction",
                "page": 1,
                "similarity_score": 0.95
            },
            chunk_index=0,
            token_count=45
        ),
        DocumentChunk(
            chunk_id="chunk_ml_1",
            document_id="machine_learning_guide",
            content="""
            Machine Learning is a subset of artificial intelligence that enables computers to learn 
            and improve from experience without being explicitly programmed. It focuses on developing 
            algorithms that can access data and use it to learn for themselves.
            """.strip(),
            metadata={
                "file_name": "ml_guide.pdf",
                "section": "overview",
                "page": 2,
                "similarity_score": 0.88
            },
            chunk_index=0,
            token_count=38
        ),
        DocumentChunk(
            chunk_id="chunk_dl_1",
            document_id="deep_learning_basics",
            content="""
            Deep Learning is a specialized subset of machine learning that uses neural networks 
            with multiple layers (hence "deep") to model and understand complex patterns in data. 
            It has been particularly successful in areas like image recognition and natural language processing.
            """.strip(),
            metadata={
                "file_name": "deep_learning.pdf",
                "section": "introduction",
                "page": 1,
                "similarity_score": 0.82
            },
            chunk_index=0,
            token_count=42
        ),
        DocumentChunk(
            chunk_id="chunk_nlp_1",
            document_id="nlp_handbook",
            content="""
            Natural Language Processing (NLP) is a field of AI that focuses on the interaction 
            between computers and human language. It involves developing algorithms and models 
            that can understand, interpret, and generate human language in a valuable way.
            """.strip(),
            metadata={
                "file_name": "nlp_handbook.pdf",
                "section": "fundamentals",
                "page": 3,
                "similarity_score": 0.79
            },
            chunk_index=0,
            token_count=40
        )
    ]


class MockVectorStorage:
    """Mock vector storage for demonstration."""
    
    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        self.stats = {
            "total_vectors": len(chunks),
            "index_type": "HNSW",
            "dimension": 768
        }
    
    def search(self, query_vector: List[float], k: int = 10, filter_metadata=None):
        """Mock search that returns chunks with simulated L2 distances."""
        # Simulate L2 distances based on similarity scores
        results = []
        for chunk in self.chunks[:k]:
            similarity = chunk.metadata.get("similarity_score", 0.5)
            # Convert similarity to L2 distance (for normalized vectors)
            l2_distance = (2 * (1 - similarity)) ** 0.5
            results.append((chunk, l2_distance))
        
        # Sort by distance (ascending)
        results.sort(key=lambda x: x[1])
        return results
    
    def get_stats(self):
        return self.stats


class MockEmbeddingService:
    """Mock embedding service for demonstration."""
    
    def __init__(self):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_dimension = 768
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate mock embedding vector."""
        # Return a mock 768-dimensional vector
        return [0.1] * 768


class MockGeminiClient:
    """Mock Gemini API client for demonstration."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        self.api_key = api_key
        self.model_name = model_name
    
    async def generate_response(self, prompt: str, max_tokens: int = 1024, 
                              temperature: float = 0.7, context: str = None) -> tuple:
        """Generate mock response."""
        # Simulate API delay
        await asyncio.sleep(0.1)
        
        # Generate contextual response based on the query
        if "artificial intelligence" in prompt.lower() or "ai" in prompt.lower():
            response = """
            Based on the provided context, Artificial Intelligence (AI) is a branch of computer science 
            focused on creating intelligent machines that can perform human-like tasks including learning, 
            reasoning, and problem-solving. AI encompasses various subfields like Machine Learning and 
            Deep Learning, which enable computers to learn from data and improve their performance over time.
            """.strip()
        elif "machine learning" in prompt.lower():
            response = """
            Machine Learning is a subset of AI that enables computers to learn and improve from experience 
            without explicit programming. It uses algorithms to access data and learn patterns, making it 
            particularly useful for tasks like prediction and classification.
            """.strip()
        elif "deep learning" in prompt.lower():
            response = """
            Deep Learning is a specialized form of machine learning that uses neural networks with multiple 
            layers to understand complex patterns in data. It has achieved remarkable success in areas like 
            image recognition and natural language processing.
            """.strip()
        else:
            response = """
            I can help answer questions about artificial intelligence, machine learning, deep learning, 
            and natural language processing based on the available documentation. Please feel free to 
            ask specific questions about these topics.
            """.strip()
        
        # Mock token usage
        token_usage = TokenUsage(
            prompt_tokens=len(prompt.split()) + (len(context.split()) if context else 0),
            completion_tokens=len(response.split()),
            total_tokens=len(prompt.split()) + (len(context.split()) if context else 0) + len(response.split())
        )
        
        return response, token_usage
    
    async def close(self):
        """Mock close method."""
        pass


async def demonstrate_rag_service():
    """Demonstrate RAG service functionality."""
    print("🚀 RAG Service Demo Starting...")
    print("=" * 60)
    
    # Create sample data
    sample_chunks = create_sample_chunks()
    print(f"📚 Created {len(sample_chunks)} sample document chunks")
    
    # Create mock services
    mock_vector_storage = MockVectorStorage(sample_chunks)
    mock_embedding_service = MockEmbeddingService()
    
    # Create RAG service with mocked dependencies
    rag_service = RAGService(
        vector_storage=mock_vector_storage,
        embedding_service=mock_embedding_service,
        gemini_api_key="demo-api-key",
        gemini_model="gemini-pro",
        max_context_chunks=3,
        similarity_threshold=0.7
    )
    
    # Replace Gemini client with mock
    rag_service.gemini_client = MockGeminiClient("demo-api-key")
    
    print("✅ RAG Service initialized successfully")
    print()
    
    # Test queries
    test_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are the applications of deep learning?",
        "Explain natural language processing"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"🔍 Query {i}: {query}")
        print("-" * 40)
        
        try:
            # Process query
            start_time = datetime.now()
            response = await rag_service.process_query(
                query=query,
                context_limit=3,
                include_sources=True
            )
            end_time = datetime.now()
            
            # Display results
            print(f"💡 Answer: {response.answer}")
            print(f"⏱️  Processing Time: {response.processing_time_ms}ms")
            print(f"🎯 Confidence Score: {response.confidence_score:.2f}")
            print(f"📊 Token Usage: {response.token_usage.total_tokens} tokens")
            print(f"📖 Sources Used: {len(response.source_chunks)} chunks")
            
            if response.source_chunks:
                print("📚 Source Documents:")
                for j, chunk in enumerate(response.source_chunks, 1):
                    file_name = chunk.metadata.get("file_name", "unknown")
                    similarity = chunk.metadata.get("similarity_score", 0.0)
                    print(f"   {j}. {file_name} (similarity: {similarity:.2f})")
            
            print()
            
        except RAGServiceError as e:
            print(f"❌ Error processing query: {e}")
            print()
    
    # Demonstrate health check
    print("🏥 Health Check")
    print("-" * 40)
    try:
        health_status = await rag_service.health_check()
        print(f"Overall Status: {health_status['status']}")
        print("Component Status:")
        for component, status in health_status['components'].items():
            status_emoji = "✅" if status['status'] == 'healthy' else "⚠️" if status['status'] == 'degraded' else "❌"
            print(f"  {status_emoji} {component}: {status['status']}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    print()
    
    # Demonstrate error handling
    print("🚨 Error Handling Demo")
    print("-" * 40)
    try:
        # Test with empty query
        await rag_service.process_query("")
    except RAGServiceError as e:
        print(f"✅ Correctly caught empty query error: {e}")
    
    try:
        # Test with very long context limit
        response = await rag_service.process_query(
            query="Test query",
            context_limit=100  # Should be clamped to max_context_chunks
        )
        print(f"✅ Context limit clamped to: {len(response.source_chunks)} chunks")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    print()
    
    # Clean up
    await rag_service.close()
    print("🧹 RAG Service closed successfully")
    print("=" * 60)
    print("✨ Demo completed!")


def demonstrate_gemini_client():
    """Demonstrate GeminiAPIClient functionality."""
    print("\n🤖 Gemini API Client Demo")
    print("-" * 40)
    
    client = GeminiAPIClient("demo-api-key", "gemini-pro")
    
    # Test prompt construction
    prompt_with_context = client._construct_prompt(
        query="What is machine learning?",
        context="Machine learning is a subset of AI that enables computers to learn from data."
    )
    
    print("📝 Prompt with context:")
    print(prompt_with_context[:200] + "..." if len(prompt_with_context) > 200 else prompt_with_context)
    print()
    
    prompt_without_context = client._construct_prompt(
        query="What is machine learning?"
    )
    
    print("📝 Prompt without context:")
    print(prompt_without_context[:200] + "..." if len(prompt_without_context) > 200 else prompt_without_context)
    print()
    
    # Test token estimation
    test_text = "This is a sample text for token estimation with multiple words and sentences."
    estimated_tokens = client._estimate_tokens(test_text)
    print(f"🔢 Token estimation for '{test_text}': {estimated_tokens} tokens")


if __name__ == "__main__":
    print("🎯 RAG Enterprise Assistant - Service Demo")
    print("This demo showcases the core RAG service functionality")
    print("with mocked dependencies for testing purposes.\n")
    
    # Run Gemini client demo
    demonstrate_gemini_client()
    
    # Run main RAG service demo
    asyncio.run(demonstrate_rag_service())