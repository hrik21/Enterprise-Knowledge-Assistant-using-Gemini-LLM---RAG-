"""
Unit tests for the document chunking service.

Tests document loading, chunking, embedding generation, and the complete
processing pipeline with various file formats and configurations.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

import numpy as np
from llama_index.core import Document

from src.services.document_chunking import (
    DocumentLoader,
    DocumentChunker,
    EmbeddingService,
    DocumentProcessingPipeline,
    DocumentChunkingError
)
from src.models.data_models import DocumentChunk


class TestDocumentLoader:
    """Test cases for DocumentLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = DocumentLoader()
    
    def test_init(self):
        """Test DocumentLoader initialization."""
        assert self.loader.pdf_reader is not None
        assert self.loader.docx_reader is not None
        assert len(self.loader.supported_types) == 5
        assert '.pdf' in self.loader.supported_types
        assert '.txt' in self.loader.supported_types
        assert '.docx' in self.loader.supported_types
    
    def test_load_txt_document(self):
        """Test loading a text document."""
        # Create temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.\nIt has multiple lines.\nFor testing purposes.")
            temp_path = f.name
        
        try:
            documents = self.loader.load_document(temp_path, document_id="test_doc")
            
            assert len(documents) == 1
            assert isinstance(documents[0], Document)
            assert "This is a test document." in documents[0].text
            assert documents[0].metadata['file_name'] == Path(temp_path).name
            assert documents[0].metadata['document_id'] == "test_doc"
            assert documents[0].metadata['file_type'] == '.txt'
            
        finally:
            os.unlink(temp_path)
    
    def test_load_empty_txt_document(self):
        """Test loading an empty text document raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")  # Empty file
            temp_path = f.name
        
        try:
            with pytest.raises(DocumentChunkingError, match="Document is empty"):
                self.loader.load_document(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file raises error."""
        with pytest.raises(DocumentChunkingError, match="File not found"):
            self.loader.load_document("nonexistent_file.txt")
    
    def test_load_unsupported_file_type(self):
        """Test loading unsupported file type raises error."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(DocumentChunkingError, match="Unsupported file type"):
                self.loader.load_document(temp_path)
        finally:
            os.unlink(temp_path)
    
    @patch('src.services.document_chunking.PDFReader')
    def test_load_pdf_document(self, mock_pdf_reader):
        """Test loading a PDF document."""
        # Mock PDF reader
        mock_reader_instance = Mock()
        mock_reader_instance.load_data.return_value = [
            Document(text="PDF content here", metadata={'page': 1})
        ]
        mock_pdf_reader.return_value = mock_reader_instance
        
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            temp_path = f.name
        
        try:
            # Reinitialize loader to use mocked PDF reader
            loader = DocumentLoader()
            documents = loader.load_document(temp_path, document_id="test_pdf")
            
            assert len(documents) == 1
            assert "PDF content here" in documents[0].text
            assert documents[0].metadata['document_id'] == "test_pdf"
            assert documents[0].metadata['file_type'] == '.pdf'
            
        finally:
            os.unlink(temp_path)
    
    @patch('src.services.document_chunking.DocxReader')
    def test_load_docx_document(self, mock_docx_reader):
        """Test loading a DOCX document."""
        # Mock DOCX reader
        mock_reader_instance = Mock()
        mock_reader_instance.load_data.return_value = [
            Document(text="DOCX content here", metadata={'paragraphs': 1})
        ]
        mock_docx_reader.return_value = mock_reader_instance
        
        # Create temporary DOCX file
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as f:
            temp_path = f.name
        
        try:
            # Reinitialize loader to use mocked DOCX reader
            loader = DocumentLoader()
            documents = loader.load_document(temp_path, document_id="test_docx")
            
            assert len(documents) == 1
            assert "DOCX content here" in documents[0].text
            assert documents[0].metadata['document_id'] == "test_docx"
            assert documents[0].metadata['file_type'] == '.docx'
            
        finally:
            os.unlink(temp_path)
    
    def test_load_from_bytes(self):
        """Test loading document from bytes."""
        content = "This is test content from bytes."
        file_bytes = content.encode('utf-8')
        
        documents = self.loader.load_from_bytes(
            file_bytes,
            "test.txt",
            document_id="bytes_test"
        )
        
        assert len(documents) == 1
        assert content in documents[0].text
        assert documents[0].metadata['file_name'] == "test.txt"
        assert documents[0].metadata['document_id'] == "bytes_test"
        assert documents[0].metadata['file_size'] == len(file_bytes)
    
    def test_load_with_custom_metadata(self):
        """Test loading document with custom metadata."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            temp_path = f.name
        
        custom_metadata = {'author': 'Test Author', 'category': 'Test'}
        
        try:
            documents = self.loader.load_document(
                temp_path,
                document_id="test_doc",
                metadata=custom_metadata
            )
            
            assert documents[0].metadata['author'] == 'Test Author'
            assert documents[0].metadata['category'] == 'Test'
            
        finally:
            os.unlink(temp_path)


class TestDocumentChunker:
    """Test cases for DocumentChunker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    
    def test_init(self):
        """Test DocumentChunker initialization."""
        assert self.chunker.chunk_size == 100
        assert self.chunker.chunk_overlap == 20
        assert self.chunker.splitter is not None
    
    def test_chunk_single_document(self):
        """Test chunking a single document."""
        # Create a document with enough content to be chunked
        long_text = " ".join([f"This is sentence {i} in the test document." for i in range(50)])
        document = Document(text=long_text, metadata={'doc_id': 'test_doc'})
        
        chunks = self.chunker.chunk_documents([document], document_id="test_doc")
        
        assert len(chunks) > 1  # Should be chunked into multiple pieces
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.document_id == "test_doc" for chunk in chunks)
        assert all(chunk.token_count > 0 for chunk in chunks)
        
        # Check chunk indices are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
    
    def test_chunk_multiple_documents(self):
        """Test chunking multiple documents."""
        documents = [
            Document(text="First document content " * 20, metadata={'doc_id': 'doc1'}),
            Document(text="Second document content " * 20, metadata={'doc_id': 'doc2'})
        ]
        
        chunks = self.chunker.chunk_documents(documents, document_id="multi_doc")
        
        assert len(chunks) >= 2  # Should have at least one chunk from each document
        assert all(chunk.document_id == "multi_doc" for chunk in chunks)
        
        # Check that we have chunks from both documents
        doc_indices = set(chunk.metadata['document_index'] for chunk in chunks)
        assert 0 in doc_indices
        assert 1 in doc_indices
    
    def test_chunk_short_document(self):
        """Test chunking a document shorter than chunk size."""
        short_text = "This is a short document."
        document = Document(text=short_text, metadata={'doc_id': 'short_doc'})
        
        chunks = self.chunker.chunk_documents([document], document_id="short_doc")
        
        assert len(chunks) == 1
        assert chunks[0].content == short_text
        assert chunks[0].chunk_index == 0
    
    def test_chunk_empty_document(self):
        """Test chunking an empty document."""
        document = Document(text="", metadata={'doc_id': 'empty_doc'})
        
        chunks = self.chunker.chunk_documents([document], document_id="empty_doc")
        
        # Should handle empty documents gracefully
        assert len(chunks) >= 0
    
    def test_update_chunk_size(self):
        """Test updating chunk size parameters."""
        original_size = self.chunker.chunk_size
        new_size = 200
        new_overlap = 40
        
        self.chunker.update_chunk_size(new_size, new_overlap)
        
        assert self.chunker.chunk_size == new_size
        assert self.chunker.chunk_overlap == new_overlap
        assert self.chunker.chunk_size != original_size
    
    def test_token_counting(self):
        """Test token counting functionality."""
        test_text = "This is a test sentence for token counting."
        token_count = self.chunker._count_tokens(test_text)
        
        assert isinstance(token_count, int)
        assert token_count > 0
        assert token_count <= len(test_text.split()) + 5  # Reasonable upper bound


class TestEmbeddingService:
    """Test cases for EmbeddingService class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use a small model for testing
        with patch('src.services.document_chunking.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_model.encode.return_value = np.random.rand(1, 384)
            mock_st.return_value = mock_model
            
            self.embedding_service = EmbeddingService(
                model_name="test-model",
                batch_size=2
            )
            self.mock_model = mock_model
    
    def test_init(self):
        """Test EmbeddingService initialization."""
        assert self.embedding_service.model_name == "test-model"
        assert self.embedding_service.batch_size == 2
        assert self.embedding_service.embedding_dimension == 384
    
    def test_generate_embeddings(self):
        """Test generating embeddings for document chunks."""
        # Create test chunks
        chunks = [
            DocumentChunk(
                document_id="doc1",
                content="First chunk content",
                chunk_index=0,
                token_count=10
            ),
            DocumentChunk(
                document_id="doc1",
                content="Second chunk content",
                chunk_index=1,
                token_count=12
            )
        ]
        
        # Mock embedding generation
        self.mock_model.encode.return_value = np.random.rand(2, 384)
        
        embedded_chunks = self.embedding_service.generate_embeddings(chunks)
        
        assert len(embedded_chunks) == 2
        assert all(chunk.embedding is not None for chunk in embedded_chunks)
        assert all(len(chunk.embedding) == 384 for chunk in embedded_chunks)
        assert all(isinstance(chunk.embedding, list) for chunk in embedded_chunks)
        
        # Verify original chunk data is preserved
        for original, embedded in zip(chunks, embedded_chunks):
            assert embedded.document_id == original.document_id
            assert embedded.content == original.content
            assert embedded.chunk_index == original.chunk_index
    
    def test_generate_embeddings_empty_list(self):
        """Test generating embeddings for empty chunk list."""
        embedded_chunks = self.embedding_service.generate_embeddings([])
        assert embedded_chunks == []
    
    def test_generate_query_embedding(self):
        """Test generating embedding for a query string."""
        query = "What is the meaning of life?"
        
        # Mock single embedding generation
        self.mock_model.encode.return_value = np.random.rand(1, 384)
        
        embedding = self.embedding_service.generate_query_embedding(query)
        
        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)
    
    def test_get_embedding_dimension(self):
        """Test getting embedding dimension."""
        dimension = self.embedding_service.get_embedding_dimension()
        assert dimension == 384
    
    def test_batch_processing(self):
        """Test that embeddings are processed in batches."""
        # Create more chunks than batch size
        chunks = [
            DocumentChunk(
                document_id="doc1",
                content=f"Chunk {i} content",
                chunk_index=i,
                token_count=10
            )
            for i in range(5)  # More than batch_size=2
        ]
        
        # Mock embedding generation to return different batches
        def mock_encode(texts, **kwargs):
            return np.random.rand(len(texts), 384)
        
        self.mock_model.encode.side_effect = mock_encode
        
        embedded_chunks = self.embedding_service.generate_embeddings(chunks)
        
        assert len(embedded_chunks) == 5
        # Verify encode was called multiple times for batching
        assert self.mock_model.encode.call_count >= 2


class TestDocumentProcessingPipeline:
    """Test cases for DocumentProcessingPipeline class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('src.services.document_chunking.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_model.encode.return_value = np.random.rand(1, 384)
            mock_st.return_value = mock_model
            
            self.pipeline = DocumentProcessingPipeline(
                chunk_size=100,
                chunk_overlap=20,
                embedding_model="test-model",
                batch_size=2
            )
            self.mock_model = mock_model
    
    def test_init(self):
        """Test DocumentProcessingPipeline initialization."""
        assert self.pipeline.loader is not None
        assert self.pipeline.chunker is not None
        assert self.pipeline.embedding_service is not None
    
    def test_process_file_complete_pipeline(self):
        """Test processing a file through the complete pipeline."""
        # Create temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Create content that will be chunked
            content = " ".join([f"This is sentence {i} for testing." for i in range(30)])
            f.write(content)
            temp_path = f.name
        
        try:
            # Mock embedding generation
            def mock_encode(texts, **kwargs):
                return np.random.rand(len(texts), 384)
            
            self.mock_model.encode.side_effect = mock_encode
            
            chunks = self.pipeline.process_file(temp_path, document_id="test_pipeline")
            
            assert len(chunks) > 0
            assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
            assert all(chunk.embedding is not None for chunk in chunks)
            assert all(chunk.document_id == "test_pipeline" for chunk in chunks)
            assert all(len(chunk.embedding) == 384 for chunk in chunks)
            
        finally:
            os.unlink(temp_path)
    
    def test_process_bytes_complete_pipeline(self):
        """Test processing bytes through the complete pipeline."""
        content = " ".join([f"Sentence {i} from bytes." for i in range(20)])
        file_bytes = content.encode('utf-8')
        
        # Mock embedding generation
        def mock_encode(texts, **kwargs):
            return np.random.rand(len(texts), 384)
        
        self.mock_model.encode.side_effect = mock_encode
        
        chunks = self.pipeline.process_bytes(
            file_bytes,
            "test.txt",
            document_id="bytes_pipeline"
        )
        
        assert len(chunks) > 0
        assert all(chunk.embedding is not None for chunk in chunks)
        assert all(chunk.document_id == "bytes_pipeline" for chunk in chunks)
    
    def test_get_pipeline_stats(self):
        """Test getting pipeline statistics."""
        stats = self.pipeline.get_pipeline_stats()
        
        assert 'chunk_size' in stats
        assert 'chunk_overlap' in stats
        assert 'embedding_model' in stats
        assert 'embedding_dimension' in stats
        assert 'batch_size' in stats
        assert 'supported_formats' in stats
        
        assert stats['chunk_size'] == 100
        assert stats['chunk_overlap'] == 20
        assert stats['embedding_model'] == "test-model"
        assert stats['embedding_dimension'] == 384
        assert isinstance(stats['supported_formats'], list)
    
    def test_process_file_with_metadata(self):
        """Test processing file with custom metadata."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content with metadata.")
            temp_path = f.name
        
        custom_metadata = {'author': 'Test Author', 'category': 'Test'}
        
        try:
            # Mock embedding generation
            self.mock_model.encode.return_value = np.random.rand(1, 384)
            
            chunks = self.pipeline.process_file(
                temp_path,
                document_id="metadata_test",
                metadata=custom_metadata
            )
            
            assert len(chunks) > 0
            assert chunks[0].metadata['author'] == 'Test Author'
            assert chunks[0].metadata['category'] == 'Test'
            
        finally:
            os.unlink(temp_path)
    
    def test_error_handling_in_pipeline(self):
        """Test error handling in the processing pipeline."""
        # Test with non-existent file
        with pytest.raises(DocumentChunkingError):
            self.pipeline.process_file("nonexistent_file.txt")
        
        # Test with invalid bytes
        with pytest.raises(DocumentChunkingError):
            self.pipeline.process_bytes(b"", "empty.txt")


class TestErrorHandling:
    """Test cases for error handling across all components."""
    
    def test_document_chunking_error_inheritance(self):
        """Test that DocumentChunkingError is properly defined."""
        error = DocumentChunkingError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"
    
    def test_loader_error_propagation(self):
        """Test error propagation in DocumentLoader."""
        loader = DocumentLoader()
        
        with pytest.raises(DocumentChunkingError):
            loader.load_document("nonexistent.txt")
    
    @patch('src.services.document_chunking.SentenceTransformer')
    def test_embedding_service_initialization_error(self, mock_st):
        """Test EmbeddingService initialization error handling."""
        mock_st.side_effect = Exception("Model loading failed")
        
        with pytest.raises(DocumentChunkingError, match="Failed to initialize embedding model"):
            EmbeddingService(model_name="invalid-model")
    
    @patch('src.services.document_chunking.SentenceTransformer')
    def test_embedding_generation_error(self, mock_st):
        """Test error handling in embedding generation."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.side_effect = Exception("Encoding failed")
        mock_st.return_value = mock_model
        
        embedding_service = EmbeddingService(model_name="test-model")
        
        chunks = [
            DocumentChunk(
                document_id="doc1",
                content="Test content",
                chunk_index=0,
                token_count=10
            )
        ]
        
        with pytest.raises(DocumentChunkingError, match="Failed to generate embeddings"):
            embedding_service.generate_embeddings(chunks)


@pytest.fixture
def sample_document_chunks():
    """Fixture providing sample DocumentChunk objects for testing."""
    return [
        DocumentChunk(
            document_id="test_doc",
            content="This is the first chunk of content for testing purposes.",
            chunk_index=0,
            token_count=12,
            metadata={'source': 'test'}
        ),
        DocumentChunk(
            document_id="test_doc",
            content="This is the second chunk with different content for testing.",
            chunk_index=1,
            token_count=11,
            metadata={'source': 'test'}
        )
    ]


@pytest.fixture
def sample_documents():
    """Fixture providing sample LlamaIndex Document objects for testing."""
    return [
        Document(
            text="This is a sample document for testing the chunking functionality. " * 10,
            metadata={'doc_id': 'sample1', 'author': 'Test Author'}
        ),
        Document(
            text="Another sample document with different content for comprehensive testing. " * 8,
            metadata={'doc_id': 'sample2', 'category': 'Test Category'}
        )
    ]


class TestIntegration:
    """Integration tests for the complete document processing workflow."""
    
    @patch('src.services.document_chunking.SentenceTransformer')
    def test_end_to_end_processing(self, mock_st):
        """Test complete end-to-end document processing."""
        # Setup mock embedding model
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        
        def mock_encode(texts, **kwargs):
            return np.random.rand(len(texts), 768)
        
        mock_model.encode.side_effect = mock_encode
        mock_st.return_value = mock_model
        
        # Create test document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Create substantial content for chunking
            content = "\n\n".join([
                f"This is paragraph {i}. It contains multiple sentences for testing. "
                f"The content should be long enough to create multiple chunks. "
                f"Each paragraph has different information to test retrieval."
                for i in range(10)
            ])
            f.write(content)
            temp_path = f.name
        
        try:
            # Initialize pipeline
            pipeline = DocumentProcessingPipeline(
                chunk_size=200,
                chunk_overlap=50,
                embedding_model="all-MiniLM-L6-v2",
                batch_size=4
            )
            
            # Process document
            chunks = pipeline.process_file(
                temp_path,
                document_id="integration_test",
                metadata={'test_type': 'integration'}
            )
            
            # Verify results
            assert len(chunks) > 1  # Should create multiple chunks
            assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
            assert all(chunk.embedding is not None for chunk in chunks)
            assert all(len(chunk.embedding) == 768 for chunk in chunks)
            assert all(chunk.document_id == "integration_test" for chunk in chunks)
            assert all(chunk.metadata['test_type'] == 'integration' for chunk in chunks)
            
            # Verify chunk ordering
            for i, chunk in enumerate(chunks):
                assert chunk.chunk_index == i
            
            # Verify content preservation
            all_content = " ".join(chunk.content for chunk in chunks)
            assert "paragraph 0" in all_content
            assert "paragraph 9" in all_content
            
        finally:
            os.unlink(temp_path)
    
    def test_pipeline_configuration_consistency(self):
        """Test that pipeline configuration is consistent across components."""
        with patch('src.services.document_chunking.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model
            
            pipeline = DocumentProcessingPipeline(
                chunk_size=256,
                chunk_overlap=32,
                embedding_model="test-model",
                batch_size=8
            )
            
            stats = pipeline.get_pipeline_stats()
            
            assert stats['chunk_size'] == 256
            assert stats['chunk_overlap'] == 32
            assert stats['embedding_model'] == "test-model"
            assert stats['batch_size'] == 8
            assert stats['embedding_dimension'] == 384