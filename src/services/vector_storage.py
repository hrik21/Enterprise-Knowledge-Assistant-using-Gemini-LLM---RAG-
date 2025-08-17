"""
FAISS Vector Storage Service for the RAG Enterprise Assistant.

This module provides a wrapper around FAISS for high-performance vector similarity search
with HNSW index configuration, persistence, and error handling.
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import faiss
from datetime import datetime

from ..models.data_models import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStorageError(Exception):
    """Custom exception for vector storage operations."""
    pass


class FAISSVectorStorage:
    """
    FAISS-based vector storage service with HNSW index configuration.
    
    Provides high-performance similarity search capabilities with persistence,
    error handling, and metadata management for document chunks.
    """
    
    def __init__(
        self,
        dimension: int = 768,
        index_type: str = "HNSW",
        storage_path: str = "data/vector_storage",
        m_hnsw: int = 16,
        ef_construction: int = 200,
        ef_search: int = 100
    ):
        """
        Initialize FAISS vector storage.
        
        Args:
            dimension: Vector embedding dimension (default: 768 for sentence-transformers)
            index_type: Type of FAISS index to use (default: "HNSW")
            storage_path: Path to store index files and metadata
            m_hnsw: Number of bi-directional links for HNSW (default: 16)
            ef_construction: Size of dynamic candidate list for construction (default: 200)
            ef_search: Size of dynamic candidate list for search (default: 100)
        """
        self.dimension = dimension
        self.index_type = index_type
        self.storage_path = Path(storage_path)
        self.m_hnsw = m_hnsw
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        
        # Create storage directory if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize index and metadata storage
        self.index: Optional[faiss.Index] = None
        self.metadata: Dict[int, DocumentChunk] = {}
        self.id_to_chunk_id: Dict[int, str] = {}
        self.chunk_id_to_id: Dict[str, int] = {}
        self.next_id = 0
        
        # File paths for persistence
        self.index_file = self.storage_path / "faiss_index.bin"
        self.metadata_file = self.storage_path / "metadata.pkl"
        self.config_file = self.storage_path / "config.json"
        
        # Initialize or load existing index
        self._initialize_index()
    
    def _initialize_index(self) -> None:
        """Initialize FAISS index with HNSW configuration."""
        try:
            if self._load_existing_index():
                logger.info("Loaded existing FAISS index from storage")
                return
            
            logger.info(f"Creating new {self.index_type} index with dimension {self.dimension}")
            
            if self.index_type == "HNSW":
                # Create HNSW index for high-performance approximate search
                # Use L2 distance for better similarity scoring
                self.index = faiss.IndexHNSWFlat(self.dimension, self.m_hnsw, faiss.METRIC_L2)
                self.index.hnsw.efConstruction = self.ef_construction
                self.index.hnsw.efSearch = self.ef_search
            else:
                # Fallback to flat index for exact search with L2 distance
                self.index = faiss.IndexFlatL2(self.dimension)
            
            # Save initial configuration
            self._save_config()
            logger.info("Successfully initialized new FAISS index")
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            raise VectorStorageError(f"Index initialization failed: {e}")
    
    def _load_existing_index(self) -> bool:
        """
        Load existing index and metadata from storage.
        
        Returns:
            bool: True if successfully loaded, False if no existing index found
        """
        try:
            if not (self.index_file.exists() and self.metadata_file.exists() and self.config_file.exists()):
                return False
            
            # Load configuration
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            # Verify configuration compatibility
            if config['dimension'] != self.dimension:
                logger.warning(f"Dimension mismatch: stored {config['dimension']}, requested {self.dimension}")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_file))
            
            # Load metadata
            with open(self.metadata_file, 'rb') as f:
                stored_data = pickle.load(f)
                self.metadata = stored_data['metadata']
                self.id_to_chunk_id = stored_data['id_to_chunk_id']
                self.chunk_id_to_id = stored_data['chunk_id_to_id']
                self.next_id = stored_data['next_id']
            
            # Update search parameters for HNSW
            if hasattr(self.index, 'hnsw'):
                self.index.hnsw.efSearch = self.ef_search
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load existing index: {e}")
            return False
    
    def _save_config(self) -> None:
        """Save index configuration to file."""
        config = {
            'dimension': self.dimension,
            'index_type': self.index_type,
            'm_hnsw': self.m_hnsw,
            'ef_construction': self.ef_construction,
            'ef_search': self.ef_search,
            'created_at': datetime.utcnow().isoformat()
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def add_vectors(self, chunks: List[DocumentChunk]) -> List[int]:
        """
        Add document chunks with their embeddings to the vector storage.
        
        Args:
            chunks: List of DocumentChunk objects with embeddings
            
        Returns:
            List[int]: Internal IDs assigned to the added vectors
            
        Raises:
            VectorStorageError: If adding vectors fails
        """
        if not chunks:
            return []
        
        try:
            # Validate that all chunks have embeddings
            embeddings = []
            chunk_ids = []
            
            for chunk in chunks:
                if chunk.embedding is None:
                    raise VectorStorageError(f"Chunk {chunk.chunk_id} has no embedding")
                
                if len(chunk.embedding) != self.dimension:
                    raise VectorStorageError(
                        f"Chunk {chunk.chunk_id} embedding dimension {len(chunk.embedding)} "
                        f"doesn't match expected {self.dimension}"
                    )
                
                embeddings.append(chunk.embedding)
                chunk_ids.append(chunk.chunk_id)
            
            # Convert to numpy array and normalize for cosine similarity
            embeddings_array = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)  # Normalize for cosine similarity with L2 distance
            
            # Assign internal IDs and store metadata
            internal_ids = []
            for i, chunk in enumerate(chunks):
                internal_id = self.next_id
                internal_ids.append(internal_id)
                
                # Store mappings and metadata
                self.id_to_chunk_id[internal_id] = chunk.chunk_id
                self.chunk_id_to_id[chunk.chunk_id] = internal_id
                self.metadata[internal_id] = chunk
                
                self.next_id += 1
            
            # Add to FAISS index
            self.index.add(embeddings_array)
            
            logger.info(f"Successfully added {len(chunks)} vectors to index")
            return internal_ids
            
        except Exception as e:
            logger.error(f"Failed to add vectors: {e}")
            raise VectorStorageError(f"Failed to add vectors: {e}")
    
    def search(
        self,
        query_vector: List[float],
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for similar vectors in the storage.
        
        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            filter_metadata: Optional metadata filters (basic implementation)
            
        Returns:
            List of tuples containing (DocumentChunk, similarity_score)
            
        Raises:
            VectorStorageError: If search fails
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty, returning no results")
            return []
        
        try:
            # Validate query vector
            if len(query_vector) != self.dimension:
                raise VectorStorageError(
                    f"Query vector dimension {len(query_vector)} doesn't match index dimension {self.dimension}"
                )
            
            # Normalize query vector for cosine similarity
            query_array = np.array([query_vector], dtype=np.float32)
            faiss.normalize_L2(query_array)
            
            # Perform search
            scores, indices = self.index.search(query_array, min(k, self.index.ntotal))
            
            # Convert results to DocumentChunk objects with scores
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid results
                    continue
                
                if idx not in self.metadata:
                    logger.warning(f"Missing metadata for index {idx}")
                    continue
                
                chunk = self.metadata[idx]
                
                # Apply basic metadata filtering if specified
                if filter_metadata and not self._matches_filter(chunk, filter_metadata):
                    continue
                
                results.append((chunk, float(score)))
            
            logger.info(f"Search returned {len(results)} results for k={k}")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise VectorStorageError(f"Search failed: {e}")
    
    def _matches_filter(self, chunk: DocumentChunk, filter_metadata: Dict[str, Any]) -> bool:
        """
        Basic metadata filtering implementation.
        
        Args:
            chunk: DocumentChunk to check
            filter_metadata: Filter criteria
            
        Returns:
            bool: True if chunk matches filter criteria
        """
        for key, value in filter_metadata.items():
            if key in chunk.metadata:
                if chunk.metadata[key] != value:
                    return False
            elif hasattr(chunk, key):
                if getattr(chunk, key) != value:
                    return False
            else:
                return False
        return True
    
    def persist_index(self) -> None:
        """
        Persist the current index and metadata to disk.
        
        Raises:
            VectorStorageError: If persistence fails
        """
        try:
            if self.index is None:
                raise VectorStorageError("No index to persist")
            
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_file))
            
            # Save metadata
            metadata_to_save = {
                'metadata': self.metadata,
                'id_to_chunk_id': self.id_to_chunk_id,
                'chunk_id_to_id': self.chunk_id_to_id,
                'next_id': self.next_id
            }
            
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(metadata_to_save, f)
            
            # Update configuration with current timestamp
            self._save_config()
            
            logger.info(f"Successfully persisted index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Failed to persist index: {e}")
            raise VectorStorageError(f"Failed to persist index: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector storage.
        
        Returns:
            Dict containing storage statistics
        """
        if self.index is None:
            return {
                'total_vectors': 0,
                'index_type': self.index_type,
                'dimension': self.dimension,
                'is_trained': False
            }
        
        return {
            'total_vectors': self.index.ntotal,
            'index_type': self.index_type,
            'dimension': self.dimension,
            'is_trained': getattr(self.index, 'is_trained', True),
            'storage_path': str(self.storage_path),
            'metadata_count': len(self.metadata)
        }
    
    def remove_vectors(self, chunk_ids: List[str]) -> int:
        """
        Remove vectors by chunk IDs (basic implementation).
        
        Note: FAISS doesn't support efficient removal, so this is a placeholder
        for future implementation with index rebuilding.
        
        Args:
            chunk_ids: List of chunk IDs to remove
            
        Returns:
            int: Number of vectors that would be removed
        """
        count = 0
        for chunk_id in chunk_ids:
            if chunk_id in self.chunk_id_to_id:
                count += 1
        
        logger.warning(f"Vector removal not implemented - would remove {count} vectors")
        return count
    
    def update_search_params(self, ef_search: Optional[int] = None) -> None:
        """
        Update search parameters for HNSW index.
        
        Args:
            ef_search: New efSearch parameter for HNSW
        """
        if ef_search is not None:
            self.ef_search = ef_search
            if self.index and hasattr(self.index, 'hnsw'):
                self.index.hnsw.efSearch = ef_search
                logger.info(f"Updated efSearch to {ef_search}")