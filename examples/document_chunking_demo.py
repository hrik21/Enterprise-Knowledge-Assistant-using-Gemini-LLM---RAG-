#!/usr/bin/env python3
"""
Demo script for the document chunking service.

This script demonstrates how to use the document processing pipeline
to load, chunk, and embed documents for the RAG system.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.document_chunking import DocumentProcessingPipeline


def create_sample_document():
    """Create a sample document for demonstration."""
    content = """
# Enterprise Knowledge Management System

## Introduction

This document provides an overview of our enterprise knowledge management system.
The system is designed to help organizations capture, organize, and retrieve
institutional knowledge effectively.

## Key Features

### Document Processing
Our system can process various document formats including PDF, Word documents,
and plain text files. The processing pipeline automatically extracts text
content and prepares it for indexing.

### Intelligent Search
The search functionality uses advanced natural language processing to understand
user queries and return relevant results. The system can handle complex queries
and provide contextual answers.

### Knowledge Organization
Documents are automatically categorized and tagged based on their content.
This helps users discover related information and navigate the knowledge base
more effectively.

## Technical Architecture

The system is built using modern technologies including:
- Python for backend processing
- FastAPI for the REST API
- FAISS for vector similarity search
- LlamaIndex for document processing
- Sentence Transformers for embeddings

## Benefits

Organizations using our knowledge management system report:
- Improved information discovery
- Reduced time spent searching for information
- Better knowledge sharing across teams
- Enhanced decision-making capabilities

## Conclusion

The enterprise knowledge management system provides a comprehensive solution
for organizations looking to better manage their institutional knowledge.
Contact our team for more information about implementation and customization options.
    """.strip()
    
    return content


def main():
    """Main demo function."""
    print("🚀 Document Chunking Service Demo")
    print("=" * 50)
    
    # Create sample document
    print("\n📄 Creating sample document...")
    sample_content = create_sample_document()
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_content)
        temp_file_path = f.name
    
    try:
        # Initialize the document processing pipeline
        print("\n🔧 Initializing document processing pipeline...")
        pipeline = DocumentProcessingPipeline(
            chunk_size=256,  # Smaller chunks for demo
            chunk_overlap=50,
            embedding_model="all-MiniLM-L6-v2",  # Fast, lightweight model
            batch_size=4
        )
        
        # Get pipeline stats
        stats = pipeline.get_pipeline_stats()
        print(f"   • Chunk size: {stats['chunk_size']} tokens")
        print(f"   • Chunk overlap: {stats['chunk_overlap']} tokens")
        print(f"   • Embedding model: {stats['embedding_model']}")
        print(f"   • Embedding dimension: {stats['embedding_dimension']}")
        print(f"   • Supported formats: {', '.join(stats['supported_formats'])}")
        
        # Process the document
        print(f"\n⚙️  Processing document: {Path(temp_file_path).name}")
        chunks = pipeline.process_file(
            temp_file_path,
            document_id="demo_document",
            metadata={
                'title': 'Enterprise Knowledge Management System',
                'category': 'documentation',
                'author': 'Demo System'
            }
        )
        
        print(f"✅ Successfully processed document into {len(chunks)} chunks")
        
        # Display chunk information
        print(f"\n📊 Chunk Analysis:")
        total_tokens = sum(chunk.token_count for chunk in chunks)
        avg_tokens = total_tokens / len(chunks) if chunks else 0
        
        print(f"   • Total chunks: {len(chunks)}")
        print(f"   • Total tokens: {total_tokens}")
        print(f"   • Average tokens per chunk: {avg_tokens:.1f}")
        print(f"   • Embedding dimension: {len(chunks[0].embedding) if chunks and chunks[0].embedding else 'N/A'}")
        
        # Display first few chunks
        print(f"\n📝 Sample Chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n   Chunk {i + 1}:")
            print(f"   • ID: {chunk.chunk_id[:8]}...")
            print(f"   • Tokens: {chunk.token_count}")
            print(f"   • Content preview: {chunk.content[:100]}...")
            if chunk.embedding:
                print(f"   • Embedding sample: [{chunk.embedding[0]:.4f}, {chunk.embedding[1]:.4f}, ...]")
        
        if len(chunks) > 3:
            print(f"\n   ... and {len(chunks) - 3} more chunks")
        
        # Test query embedding
        print(f"\n🔍 Testing query embedding...")
        query = "What are the key features of the knowledge management system?"
        query_embedding = pipeline.embedding_service.generate_query_embedding(query)
        print(f"   • Query: {query}")
        print(f"   • Query embedding dimension: {len(query_embedding)}")
        print(f"   • Query embedding sample: [{query_embedding[0]:.4f}, {query_embedding[1]:.4f}, ...]")
        
        # Calculate similarity with first chunk (simple demonstration)
        if chunks and chunks[0].embedding:
            import numpy as np
            
            chunk_emb = np.array(chunks[0].embedding)
            query_emb = np.array(query_embedding)
            
            # Cosine similarity
            similarity = np.dot(chunk_emb, query_emb) / (np.linalg.norm(chunk_emb) * np.linalg.norm(query_emb))
            print(f"   • Similarity with first chunk: {similarity:.4f}")
        
        print(f"\n✨ Demo completed successfully!")
        print(f"\nThe document chunking service is ready for integration with:")
        print(f"   • Vector storage (FAISS)")
        print(f"   • RAG query processing")
        print(f"   • Document ingestion pipelines")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        return 1
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
    
    return 0


if __name__ == "__main__":
    exit(main())