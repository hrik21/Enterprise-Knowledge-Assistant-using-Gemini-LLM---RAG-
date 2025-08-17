"""
Services module for the RAG Enterprise Assistant.

This module contains service classes that implement the core business logic
for vector storage, document processing, and RAG operations.
"""

from .vector_storage import FAISSVectorStorage, VectorStorageError

__all__ = [
    'FAISSVectorStorage',
    'VectorStorageError'
]