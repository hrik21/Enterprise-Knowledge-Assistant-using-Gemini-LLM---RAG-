"""
Unit tests for FAISS Vector Storage Service.

Tests cover vector operations, persistence, error handling, and edge cases
with mock data to ensure reliable vector storage functionality.
"""

import json
import pickle
import tempfile
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import List

from src.services.vector_storage import FAISSVectorStorage, VectorStorageError
from src.models.data_models import DocumentChunk


class TestFAISSVectorStorage:
    """Test suite for FAISS Vector Storage Service."""
    
    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def sample_chunks(self) -> List[DocumentChunk]:
        """Create sample document chunks with embeddings for testing."""
        chunks = []
        for i in range(5):
            # Create normalized random embeddings
            embedding = np.random.randn(768).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            
            chunk = DocumentChunk(
                chunk_id=f"chunk_{i}",
                document_id=f"doc_{i // 2}",  # 2-3 chunks per document
                content=f"This is test content for chunk {i}",
                metadata={"source": f"test_doc_{i}.txt", "page": i + 1},
                embedding=embedding.tolist(),
                chunk_index=i,
                token_count=50 + i * 10
            )
            chunks.append(chunk)
        
        return chunks
    
    @pytest.fixture
    def vector_storage(self, temp_storage_path):
        """Create FAISS vector storage instance for testing."""
        return FAISSVectorStorage(
            dimension=768,
            storage_path=temp_storage_path,
            m_hnsw=8,  # Smaller for testing
            ef_construction=100,
            ef_search=50
        )
    
    def test_initialization_new_index(self, temp_storage_path):
        """Test initialization of new FAISS index."""
        storage = FAISSVectorStorage(storage_path=temp_storage_path)
        
        assert storage.dimension == 768
        assert storage.index_type == "HNSW"
        assert storage.index is not None
        assert storage.index.ntotal == 0
        assert len(storage.metadata) == 0
        assert storage.next_id == 0
        
        # Check that storage directory was created
        storage_path = Path(temp_storage_path)
        assert storage_path.exists()
        assert (storage_path / "config.json").exists()
    
    def test_initialization_with_custom_params(self, temp_storage_path):
        """Test initialization with custom parameters."""
        storage = FAISSVectorStorage(
            dimension=512,
            index_type="HNSW",
            storage_path=temp_storage_path,
            m_hnsw=32,
            ef_construction=400,
            ef_search=200
        )
        
        assert storage.dimension == 512
        assert storage.m_hnsw == 32
        assert storage.ef_construction == 400
        assert storage.ef_search == 200
    
    def test_add_vectors_success(self, vector_storage, sample_chunks):
        """Test successful addition of vectors."""
        # Add first 3 chunks
        chunks_to_add = sample_chunks[:3]
        internal_ids = vector_storage.add_vectors(chunks_to_add)
        
        assert len(internal_ids) == 3
        assert vector_storage.index.ntotal == 3
        assert len(vector_storage.metadata) == 3
        assert vector_storage.next_id == 3
        
        # Verify metadata storage
        for i, chunk in enumerate(chunks_to_add):
            internal_id = internal_ids[i]
            assert vector_storage.id_to_chunk_id[internal_id] == chunk.chunk_id
            assert vector_storage.chunk_id_to_id[chunk.chunk_id] == internal_id
            assert vector_storage.metadata[internal_id] == chunk
    
    def test_add_vectors_empty_list(self, vector_storage):
        """Test adding empty list of vectors."""
        result = vector_storage.add_vectors([])
        assert result == []
        assert vector_storage.index.ntotal == 0
    
    def test_add_vectors_missing_embedding(self, vector_storage):
        """Test error when chunk has no embedding."""
        chunk = DocumentChunk(
            chunk_id="test_chunk",
            document_id="test_doc",
            content="Test content",
            embedding=None,  # Missing embedding
            chunk_index=0,
            token_count=10
        )
        
        with pytest.raises(VectorStorageError, match="has no embedding"):
            vector_storage.add_vectors([chunk])
    
    def test_add_vectors_wrong_dimension(self, vector_storage):
        """Test error when embedding has wrong dimension."""
        # Create chunk with wrong dimension embedding directly (bypass Pydantic validation)
        chunk = DocumentChunk(
            chunk_id="test_chunk",
            document_id="test_doc",
            content="Test content",
            embedding=[0.1] * 768,  # Start with correct dimension
            chunk_index=0,
            token_count=10
        )
        # Modify embedding after creation to test vector storage validation
        chunk.embedding = [1.0, 2.0, 3.0]  # Wrong dimension
        
        with pytest.raises(VectorStorageError, match="embedding dimension"):
            vector_storage.add_vectors([chunk])
    
    def test_search_success(self, vector_storage, sample_chunks):
        """Test successful vector search."""
        # Add vectors first
        vector_storage.add_vectors(sample_chunks)
        
        # Use first chunk's embedding as query
        query_vector = sample_chunks[0].embedding
        results = vector_storage.search(query_vector, k=3)
        
        assert len(results) <= 3
        assert len(results) > 0
        
        # Check result format
        for chunk, score in results:
            assert isinstance(chunk, DocumentChunk)
            assert isinstance(score, float)
            # FAISS returns inner product scores, which can be > 1.0 for normalized vectors
            assert score >= 0.0
        
        # First result should be the same chunk (lowest distance for L2)
        best_chunk, best_score = results[0]
        assert best_chunk.chunk_id == sample_chunks[0].chunk_id
        assert best_score < 0.01  # Should be very low distance (high similarity) for L2
    
    def test_search_empty_index(self, vector_storage):
        """Test search on empty index."""
        query_vector = [0.1] * 768
        results = vector_storage.search(query_vector, k=5)
        assert results == []
    
    def test_search_wrong_dimension(self, vector_storage, sample_chunks):
        """Test search with wrong query vector dimension."""
        vector_storage.add_vectors(sample_chunks)
        
        query_vector = [1.0, 2.0, 3.0]  # Wrong dimension
        with pytest.raises(VectorStorageError, match="Query vector dimension"):
            vector_storage.search(query_vector, k=5)
    
    def test_search_with_metadata_filter(self, vector_storage, sample_chunks):
        """Test search with metadata filtering."""
        vector_storage.add_vectors(sample_chunks)
        
        query_vector = sample_chunks[0].embedding
        filter_metadata = {"source": "test_doc_0.txt"}
        
        results = vector_storage.search(query_vector, k=5, filter_metadata=filter_metadata)
        
        # Should only return chunks matching the filter
        for chunk, score in results:
            assert chunk.metadata["source"] == "test_doc_0.txt"
    
    def test_persist_and_load_index(self, temp_storage_path, sample_chunks):
        """Test index persistence and loading."""
        # Create and populate storage
        storage1 = FAISSVectorStorage(storage_path=temp_storage_path)
        storage1.add_vectors(sample_chunks)
        original_stats = storage1.get_stats()
        
        # Persist the index
        storage1.persist_index()
        
        # Create new storage instance (should load existing index)
        storage2 = FAISSVectorStorage(storage_path=temp_storage_path)
        loaded_stats = storage2.get_stats()
        
        # Verify loaded data matches original
        assert loaded_stats['total_vectors'] == original_stats['total_vectors']
        assert loaded_stats['dimension'] == original_stats['dimension']
        assert len(storage2.metadata) == len(storage1.metadata)
        
        # Test search on loaded index
        query_vector = sample_chunks[0].embedding
        results = storage2.search(query_vector, k=3)
        assert len(results) > 0
    
    def test_persist_empty_index(self, vector_storage):
        """Test persisting empty index."""
        with pytest.raises(VectorStorageError, match="No index to persist"):
            vector_storage.index = None
            vector_storage.persist_index()
    
    @patch('faiss.write_index')
    def test_persist_index_faiss_error(self, mock_write_index, vector_storage, sample_chunks):
        """Test error handling during index persistence."""
        vector_storage.add_vectors(sample_chunks)
        mock_write_index.side_effect = Exception("FAISS write error")
        
        with pytest.raises(VectorStorageError, match="Failed to persist index"):
            vector_storage.persist_index()
    
    def test_get_stats(self, vector_storage, sample_chunks):
        """Test getting storage statistics."""
        # Test empty index stats
        empty_stats = vector_storage.get_stats()
        assert empty_stats['total_vectors'] == 0
        assert empty_stats['index_type'] == "HNSW"
        assert empty_stats['dimension'] == 768
        
        # Add vectors and test populated stats
        vector_storage.add_vectors(sample_chunks)
        populated_stats = vector_storage.get_stats()
        assert populated_stats['total_vectors'] == len(sample_chunks)
        assert populated_stats['metadata_count'] == len(sample_chunks)
    
    def test_remove_vectors_placeholder(self, vector_storage, sample_chunks):
        """Test vector removal placeholder functionality."""
        vector_storage.add_vectors(sample_chunks)
        
        chunk_ids_to_remove = [sample_chunks[0].chunk_id, sample_chunks[1].chunk_id]
        removed_count = vector_storage.remove_vectors(chunk_ids_to_remove)
        
        # Should return count but not actually remove (placeholder implementation)
        assert removed_count == 2
        assert vector_storage.index.ntotal == len(sample_chunks)  # Still all there
    
    def test_update_search_params(self, vector_storage):
        """Test updating search parameters."""
        original_ef_search = vector_storage.ef_search
        new_ef_search = 150
        
        vector_storage.update_search_params(ef_search=new_ef_search)
        
        assert vector_storage.ef_search == new_ef_search
        if hasattr(vector_storage.index, 'hnsw'):
            assert vector_storage.index.hnsw.efSearch == new_ef_search
    
    def test_config_file_creation(self, temp_storage_path):
        """Test that configuration file is created correctly."""
        storage = FAISSVectorStorage(storage_path=temp_storage_path)
        
        config_file = Path(temp_storage_path) / "config.json"
        assert config_file.exists()
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        assert config['dimension'] == 768
        assert config['index_type'] == "HNSW"
        assert 'created_at' in config
    
    def test_dimension_mismatch_on_load(self, temp_storage_path, sample_chunks):
        """Test handling of dimension mismatch when loading existing index."""
        # Create storage with dimension 768
        storage1 = FAISSVectorStorage(dimension=768, storage_path=temp_storage_path)
        storage1.add_vectors(sample_chunks)
        storage1.persist_index()
        
        # Try to create storage with different dimension
        storage2 = FAISSVectorStorage(dimension=512, storage_path=temp_storage_path)
        
        # Should create new index instead of loading incompatible one
        assert storage2.index.ntotal == 0
        assert storage2.dimension == 512
    
    @patch('faiss.IndexHNSWFlat')
    def test_faiss_initialization_error(self, mock_index_class, temp_storage_path):
        """Test error handling during FAISS index initialization."""
        mock_index_class.side_effect = Exception("FAISS initialization error")
        
        with pytest.raises(VectorStorageError, match="Index initialization failed"):
            FAISSVectorStorage(storage_path=temp_storage_path)
    
    def test_flat_index_fallback(self, temp_storage_path):
        """Test fallback to flat index for non-HNSW index type."""
        storage = FAISSVectorStorage(
            storage_path=temp_storage_path,
            index_type="FLAT"
        )
        
        # Should create IndexFlatIP instead of HNSW
        assert storage.index is not None
        assert not hasattr(storage.index, 'hnsw')
    
    def test_search_with_k_larger_than_index(self, vector_storage, sample_chunks):
        """Test search with k larger than number of vectors in index."""
        vector_storage.add_vectors(sample_chunks[:2])  # Only 2 vectors
        
        query_vector = sample_chunks[0].embedding
        results = vector_storage.search(query_vector, k=10)  # Request 10
        
        # Should return only available vectors
        assert len(results) <= 2
    
    def test_matches_filter_method(self, vector_storage):
        """Test the _matches_filter method."""
        chunk = DocumentChunk(
            chunk_id="test_chunk",
            document_id="test_doc",
            content="Test content",
            metadata={"source": "test.txt", "category": "technical"},
            embedding=[0.1] * 768,
            chunk_index=0,
            token_count=10
        )
        
        # Test matching filter
        assert vector_storage._matches_filter(chunk, {"source": "test.txt"})
        assert vector_storage._matches_filter(chunk, {"category": "technical"})
        assert vector_storage._matches_filter(chunk, {"document_id": "test_doc"})
        
        # Test non-matching filter
        assert not vector_storage._matches_filter(chunk, {"source": "other.txt"})
        assert not vector_storage._matches_filter(chunk, {"nonexistent": "value"})
        
        # Test multiple criteria
        assert vector_storage._matches_filter(chunk, {
            "source": "test.txt",
            "category": "technical"
        })
        assert not vector_storage._matches_filter(chunk, {
            "source": "test.txt",
            "category": "other"
        })


class TestVectorStorageIntegration:
    """Integration tests for vector storage with realistic scenarios."""
    
    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def realistic_chunks(self) -> List[DocumentChunk]:
        """Create realistic document chunks for integration testing."""
        chunks = []
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand text.",
            "Computer vision enables machines to interpret visual information.",
            "Reinforcement learning trains agents through rewards and penalties."
        ]
        
        for i, content in enumerate(documents):
            # Create more realistic embeddings (still random but consistent)
            np.random.seed(i)  # Consistent embeddings for same content
            embedding = np.random.randn(768).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            
            chunk = DocumentChunk(
                chunk_id=f"realistic_chunk_{i}",
                document_id=f"ai_doc_{i // 2}",
                content=content,
                metadata={
                    "topic": "artificial_intelligence",
                    "difficulty": "beginner" if i < 3 else "intermediate",
                    "source": f"ai_textbook_page_{i + 10}.pdf"
                },
                embedding=embedding.tolist(),
                chunk_index=i,
                token_count=len(content.split())
            )
            chunks.append(chunk)
        
        return chunks
    
    def test_end_to_end_workflow(self, temp_storage_path, realistic_chunks):
        """Test complete workflow: add, search, persist, reload, search again."""
        # Initialize storage
        storage = FAISSVectorStorage(storage_path=temp_storage_path)
        
        # Add vectors
        internal_ids = storage.add_vectors(realistic_chunks)
        assert len(internal_ids) == len(realistic_chunks)
        
        # Perform search
        query_vector = realistic_chunks[0].embedding
        results = storage.search(query_vector, k=3)
        assert len(results) == 3
        
        # Persist index
        storage.persist_index()
        
        # Create new storage instance (loads from disk)
        new_storage = FAISSVectorStorage(storage_path=temp_storage_path)
        
        # Search again with new instance
        new_results = new_storage.search(query_vector, k=3)
        assert len(new_results) == 3
        
        # Results should be similar (same top result)
        assert results[0][0].chunk_id == new_results[0][0].chunk_id
    
    def test_large_batch_operations(self, temp_storage_path):
        """Test performance with larger batches of vectors."""
        storage = FAISSVectorStorage(storage_path=temp_storage_path)
        
        # Create larger batch of chunks
        large_batch = []
        for i in range(100):
            embedding = np.random.randn(768).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            
            chunk = DocumentChunk(
                chunk_id=f"batch_chunk_{i}",
                document_id=f"batch_doc_{i // 10}",
                content=f"Batch content {i}",
                metadata={"batch": i // 20},
                embedding=embedding.tolist(),
                chunk_index=i % 10,
                token_count=20
            )
            large_batch.append(chunk)
        
        # Add all chunks
        internal_ids = storage.add_vectors(large_batch)
        assert len(internal_ids) == 100
        assert storage.index.ntotal == 100
        
        # Test search performance
        query_vector = large_batch[0].embedding
        results = storage.search(query_vector, k=10)
        assert len(results) == 10
        
        # Test with metadata filtering
        filtered_results = storage.search(
            query_vector, 
            k=20, 
            filter_metadata={"batch": 0}
        )
        # Should return fewer results due to filtering
        assert len(filtered_results) <= 20