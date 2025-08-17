#!/usr/bin/env python3
"""
Demo script for FAISS Vector Storage Service.

This script demonstrates how to use the FAISSVectorStorage service
to add document chunks, perform similarity search, and persist the index.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.vector_storage import FAISSVectorStorage
from src.models.data_models import DocumentChunk


def create_sample_chunks():
    """Create sample document chunks with embeddings for demonstration."""
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Deep learning uses neural networks with multiple layers to learn complex patterns.",
        "Natural language processing helps computers understand and generate human language.",
        "Computer vision enables machines to interpret and analyze visual information.",
        "Reinforcement learning trains agents through rewards and penalties in an environment."
    ]
    
    chunks = []
    for i, content in enumerate(documents):
        # Create realistic embeddings (normally these would come from a model like sentence-transformers)
        np.random.seed(i)  # Consistent embeddings for demo
        embedding = np.random.randn(768).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        
        chunk = DocumentChunk(
            chunk_id=f"demo_chunk_{i}",
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


def main():
    """Main demonstration function."""
    print("🚀 FAISS Vector Storage Demo")
    print("=" * 50)
    
    # Create temporary storage directory
    storage_path = "demo_vector_storage"
    
    try:
        # Initialize vector storage
        print("1. Initializing FAISS Vector Storage...")
        vector_storage = FAISSVectorStorage(
            dimension=768,
            storage_path=storage_path,
            m_hnsw=16,
            ef_construction=200,
            ef_search=100
        )
        
        # Create sample chunks
        print("2. Creating sample document chunks...")
        chunks = create_sample_chunks()
        print(f"   Created {len(chunks)} document chunks")
        
        # Add vectors to storage
        print("3. Adding vectors to FAISS index...")
        internal_ids = vector_storage.add_vectors(chunks)
        print(f"   Added {len(internal_ids)} vectors to index")
        
        # Display storage statistics
        stats = vector_storage.get_stats()
        print("4. Storage Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Perform similarity search
        print("5. Performing similarity search...")
        query_text = "What is machine learning?"
        
        # Use the first chunk's embedding as a query (in practice, you'd embed the query text)
        query_vector = chunks[0].embedding
        results = vector_storage.search(query_vector, k=3)
        
        print(f"   Query: '{query_text}'")
        print(f"   Found {len(results)} similar chunks:")
        
        for i, (chunk, score) in enumerate(results, 1):
            print(f"   {i}. Score: {score:.4f}")
            print(f"      Content: {chunk.content[:80]}...")
            print(f"      Source: {chunk.metadata.get('source', 'Unknown')}")
            print()
        
        # Test metadata filtering
        print("6. Testing metadata filtering...")
        filtered_results = vector_storage.search(
            query_vector, 
            k=5, 
            filter_metadata={"difficulty": "beginner"}
        )
        print(f"   Found {len(filtered_results)} chunks with difficulty='beginner'")
        
        # Persist the index
        print("7. Persisting index to disk...")
        vector_storage.persist_index()
        print("   Index saved successfully")
        
        # Test loading from disk
        print("8. Testing index loading...")
        new_storage = FAISSVectorStorage(storage_path=storage_path)
        new_stats = new_storage.get_stats()
        print(f"   Loaded index with {new_stats['total_vectors']} vectors")
        
        # Verify search works on loaded index
        new_results = new_storage.search(query_vector, k=2)
        print(f"   Search on loaded index returned {len(new_results)} results")
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        return 1
    
    finally:
        # Cleanup demo files
        import shutil
        if os.path.exists(storage_path):
            shutil.rmtree(storage_path)
            print(f"🧹 Cleaned up demo files in {storage_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())