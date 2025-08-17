"""
Document Chunking Service for the RAG Enterprise Assistant.

This module provides document loading, chunking, and embedding generation
capabilities using LlamaIndex and sentence-transformers.
"""

import logging
import mimetypes
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from io import BytesIO

import numpy as np
from sentence_transformers import SentenceTransformer
from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader, DocxReader
from llama_index.core.schema import BaseNode, TextNode

from ..models.data_models import DocumentChunk, DocumentMetadata, ProcessingStatus

logger = logging.getLogger(__name__)


class DocumentChunkingError(Exception):
    """Custom exception for document chunking operations."""
    pass


class DocumentLoader:
    """
    Document loader for various file formats including PDF, TXT, and DOCX.
    
    Provides unified interface for loading different document types and
    extracting text content with metadata preservation.
    """
    
    def __init__(self):
        """Initialize document loader with supported readers."""
        self.pdf_reader = PDFReader()
        self.docx_reader = DocxReader()
        
        # Supported file types mapping
        self.supported_types = {
            '.pdf': self._load_pdf,
            '.txt': self._load_txt,
            '.docx': self._load_docx,
            '.doc': self._load_docx,  # Try docx reader for .doc files
            '.md': self._load_txt,    # Treat markdown as text
        }
    
    def load_document(
        self,
        file_path: Union[str, Path],
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Load a document from file path and return LlamaIndex Document objects.
        
        Args:
            file_path: Path to the document file
            document_id: Optional document ID for tracking
            metadata: Optional additional metadata
            
        Returns:
            List[Document]: LlamaIndex Document objects
            
        Raises:
            DocumentChunkingError: If loading fails
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise DocumentChunkingError(f"File not found: {file_path}")
            
            # Determine file type
            file_extension = file_path.suffix.lower()
            if file_extension not in self.supported_types:
                # Try to determine from MIME type
                mime_type, _ = mimetypes.guess_type(str(file_path))
                if mime_type == 'application/pdf':
                    file_extension = '.pdf'
                elif mime_type in ['text/plain', 'text/markdown']:
                    file_extension = '.txt'
                elif mime_type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                    file_extension = '.docx'
                else:
                    raise DocumentChunkingError(f"Unsupported file type: {file_extension}")
            
            # Load document using appropriate reader
            loader_func = self.supported_types[file_extension]
            documents = loader_func(file_path)
            
            # Add metadata to documents
            for doc in documents:
                doc_metadata = {
                    'file_path': str(file_path),
                    'file_name': file_path.name,
                    'file_size': file_path.stat().st_size,
                    'file_type': file_extension,
                    'document_id': document_id or file_path.stem,
                }
                
                if metadata:
                    doc_metadata.update(metadata)
                
                doc.metadata.update(doc_metadata)
            
            logger.info(f"Successfully loaded {len(documents)} document(s) from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {e}")
            raise DocumentChunkingError(f"Failed to load document: {e}")
    
    def _load_pdf(self, file_path: Path) -> List[Document]:
        """Load PDF document using PDFReader."""
        try:
            documents = self.pdf_reader.load_data(file=str(file_path))
            return documents
        except Exception as e:
            raise DocumentChunkingError(f"Failed to load PDF: {e}")
    
    def _load_txt(self, file_path: Path) -> List[Document]:
        """Load text document (TXT, MD) as plain text."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                raise DocumentChunkingError("Document is empty")
            
            document = Document(text=content)
            return [document]
            
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                document = Document(text=content)
                return [document]
            except Exception as e:
                raise DocumentChunkingError(f"Failed to decode text file: {e}")
        except Exception as e:
            raise DocumentChunkingError(f"Failed to load text file: {e}")
    
    def _load_docx(self, file_path: Path) -> List[Document]:
        """Load DOCX document using DocxReader."""
        try:
            documents = self.docx_reader.load_data(file=str(file_path))
            return documents
        except Exception as e:
            raise DocumentChunkingError(f"Failed to load DOCX: {e}")
    
    def load_from_bytes(
        self,
        file_bytes: bytes,
        file_name: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Load document from bytes data.
        
        Args:
            file_bytes: Document content as bytes
            file_name: Original file name for type detection
            document_id: Optional document ID
            metadata: Optional additional metadata
            
        Returns:
            List[Document]: LlamaIndex Document objects
        """
        try:
            # Create temporary file for processing
            import tempfile
            
            file_extension = Path(file_name).suffix.lower()
            
            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
                temp_file.write(file_bytes)
                temp_path = Path(temp_file.name)
            
            try:
                # Load using file path method
                documents = self.load_document(
                    temp_path,
                    document_id=document_id,
                    metadata=metadata
                )
                
                # Update metadata with original file name
                for doc in documents:
                    doc.metadata['file_name'] = file_name
                    doc.metadata['file_size'] = len(file_bytes)
                
                return documents
                
            finally:
                # Clean up temporary file
                if temp_path.exists():
                    temp_path.unlink()
                    
        except Exception as e:
            logger.error(f"Failed to load document from bytes: {e}")
            raise DocumentChunkingError(f"Failed to load document from bytes: {e}")


class DocumentChunker:
    """
    Document chunking service using LlamaIndex SentenceSplitter.
    
    Provides configurable chunking with overlap and token counting
    for optimal retrieval performance.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separator: str = " ",
        paragraph_separator: str = "\n\n",
        secondary_chunking_regex: str = "[^,.;。？！]+[,.;。？！]?"
    ):
        """
        Initialize document chunker with configurable parameters.
        
        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
            separator: Primary separator for splitting
            paragraph_separator: Separator for paragraph boundaries
            secondary_chunking_regex: Regex for secondary chunking
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.paragraph_separator = paragraph_separator
        self.secondary_chunking_regex = secondary_chunking_regex
        
        # Initialize LlamaIndex sentence splitter
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
            paragraph_separator=paragraph_separator,
            secondary_chunking_regex=secondary_chunking_regex
        )
        
        logger.info(f"Initialized DocumentChunker with chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def chunk_documents(
        self,
        documents: List[Document],
        document_id: Optional[str] = None
    ) -> List[DocumentChunk]:
        """
        Chunk documents into smaller pieces for embedding and retrieval.
        
        Args:
            documents: List of LlamaIndex Document objects
            document_id: Optional document ID for tracking
            
        Returns:
            List[DocumentChunk]: List of document chunks
            
        Raises:
            DocumentChunkingError: If chunking fails
        """
        try:
            all_chunks = []
            
            for doc_idx, document in enumerate(documents):
                # Split document into nodes using LlamaIndex
                nodes = self.splitter.get_nodes_from_documents([document])
                
                # Convert nodes to DocumentChunk objects
                for chunk_idx, node in enumerate(nodes):
                    # Skip empty nodes
                    if not node.text or not node.text.strip():
                        continue
                        
                    # Extract metadata from document and node
                    chunk_metadata = document.metadata.copy()
                    chunk_metadata.update({
                        'document_index': doc_idx,
                        'source_document_id': document_id or document.metadata.get('document_id'),
                        'node_id': node.node_id,
                        'start_char_idx': getattr(node, 'start_char_idx', None),
                        'end_char_idx': getattr(node, 'end_char_idx', None),
                    })
                    
                    # Count tokens in the chunk
                    token_count = self._count_tokens(node.text)
                    
                    # Create DocumentChunk
                    chunk = DocumentChunk(
                        document_id=document_id or document.metadata.get('document_id', f"doc_{doc_idx}"),
                        content=node.text.strip(),
                        metadata=chunk_metadata,
                        chunk_index=chunk_idx,
                        token_count=token_count
                    )
                    
                    all_chunks.append(chunk)
            
            logger.info(f"Successfully chunked {len(documents)} document(s) into {len(all_chunks)} chunks")
            return all_chunks
            
        except Exception as e:
            logger.error(f"Failed to chunk documents: {e}")
            raise DocumentChunkingError(f"Failed to chunk documents: {e}")
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using simple word count estimation.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            int: Number of tokens
        """
        # Simple word count estimation (tokens are roughly 0.75 * words)
        word_count = len(text.split())
        return max(1, int(word_count * 0.75))
    
    def update_chunk_size(self, chunk_size: int, chunk_overlap: Optional[int] = None) -> None:
        """
        Update chunking parameters.
        
        Args:
            chunk_size: New chunk size
            chunk_overlap: New overlap size (optional)
        """
        self.chunk_size = chunk_size
        if chunk_overlap is not None:
            self.chunk_overlap = chunk_overlap
        
        # Recreate splitter with new parameters
        self.splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=self.separator,
            paragraph_separator=self.paragraph_separator,
            secondary_chunking_regex=self.secondary_chunking_regex
        )
        
        logger.info(f"Updated chunk parameters: size={chunk_size}, overlap={self.chunk_overlap}")


class EmbeddingService:
    """
    Embedding generation service using sentence-transformers.
    
    Provides high-quality embeddings for document chunks with
    configurable models and batch processing.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        Initialize embedding service with sentence-transformers model.
        
        Args:
            model_name: Name of the sentence-transformers model
            device: Device to use ('cpu', 'cuda', etc.)
            batch_size: Batch size for embedding generation
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"Initialized EmbeddingService with model {model_name}, dimension {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model {model_name}: {e}")
            raise DocumentChunkingError(f"Failed to initialize embedding model: {e}")
    
    def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Generate embeddings for document chunks.
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            List[DocumentChunk]: Chunks with embeddings added
            
        Raises:
            DocumentChunkingError: If embedding generation fails
        """
        if not chunks:
            return []
        
        try:
            # Extract text content from chunks
            texts = [chunk.content for chunk in chunks]
            
            # Generate embeddings in batches
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True  # Normalize for cosine similarity
                )
                all_embeddings.extend(batch_embeddings)
            
            # Add embeddings to chunks
            embedded_chunks = []
            for chunk, embedding in zip(chunks, all_embeddings):
                # Create new chunk with embedding
                embedded_chunk = DocumentChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                    embedding=embedding.tolist(),  # Convert numpy array to list
                    created_at=chunk.created_at,
                    chunk_index=chunk.chunk_index,
                    token_count=chunk.token_count
                )
                embedded_chunks.append(embedded_chunk)
            
            logger.info(f"Successfully generated embeddings for {len(chunks)} chunks")
            return embedded_chunks
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise DocumentChunkingError(f"Failed to generate embeddings: {e}")
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a query string.
        
        Args:
            query: Query text
            
        Returns:
            List[float]: Query embedding vector
            
        Raises:
            DocumentChunkingError: If embedding generation fails
        """
        try:
            embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0]
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise DocumentChunkingError(f"Failed to generate query embedding: {e}")
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension of the current model."""
        return self.embedding_dimension


class DocumentProcessingPipeline:
    """
    Complete document processing pipeline combining loading, chunking, and embedding.
    
    Provides a unified interface for processing documents from various sources
    into embedded chunks ready for vector storage.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embedding_model: str = "all-MiniLM-L6-v2",
        batch_size: int = 32
    ):
        """
        Initialize document processing pipeline.
        
        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Number of overlapping tokens
            embedding_model: Sentence-transformers model name
            batch_size: Batch size for embedding generation
        """
        self.loader = DocumentLoader()
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embedding_service = EmbeddingService(
            model_name=embedding_model,
            batch_size=batch_size
        )
        
        logger.info("Initialized DocumentProcessingPipeline")
    
    def process_file(
        self,
        file_path: Union[str, Path],
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Process a single file through the complete pipeline.
        
        Args:
            file_path: Path to the document file
            document_id: Optional document ID
            metadata: Optional additional metadata
            
        Returns:
            List[DocumentChunk]: Processed chunks with embeddings
        """
        try:
            # Load document
            documents = self.loader.load_document(file_path, document_id, metadata)
            
            # Chunk documents
            chunks = self.chunker.chunk_documents(documents, document_id)
            
            # Generate embeddings
            embedded_chunks = self.embedding_service.generate_embeddings(chunks)
            
            logger.info(f"Successfully processed file {file_path} into {len(embedded_chunks)} embedded chunks")
            return embedded_chunks
            
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            raise DocumentChunkingError(f"Failed to process file: {e}")
    
    def process_bytes(
        self,
        file_bytes: bytes,
        file_name: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Process document from bytes through the complete pipeline.
        
        Args:
            file_bytes: Document content as bytes
            file_name: Original file name
            document_id: Optional document ID
            metadata: Optional additional metadata
            
        Returns:
            List[DocumentChunk]: Processed chunks with embeddings
        """
        try:
            # Load document from bytes
            documents = self.loader.load_from_bytes(file_bytes, file_name, document_id, metadata)
            
            # Chunk documents
            chunks = self.chunker.chunk_documents(documents, document_id)
            
            # Generate embeddings
            embedded_chunks = self.embedding_service.generate_embeddings(chunks)
            
            logger.info(f"Successfully processed bytes for {file_name} into {len(embedded_chunks)} embedded chunks")
            return embedded_chunks
            
        except Exception as e:
            logger.error(f"Failed to process bytes for {file_name}: {e}")
            raise DocumentChunkingError(f"Failed to process bytes: {e}")
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the processing pipeline.
        
        Returns:
            Dict containing pipeline configuration and stats
        """
        return {
            'chunk_size': self.chunker.chunk_size,
            'chunk_overlap': self.chunker.chunk_overlap,
            'embedding_model': self.embedding_service.model_name,
            'embedding_dimension': self.embedding_service.embedding_dimension,
            'batch_size': self.embedding_service.batch_size,
            'supported_formats': list(self.loader.supported_types.keys())
        }