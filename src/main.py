"""
Main entry point for RAG Enterprise Assistant.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import settings


def main():
    """Main application entry point."""
    print(f"RAG Enterprise Assistant")
    print(f"Environment: {settings.environment}")
    print(f"Debug mode: {settings.debug}")
    print(f"Configuration loaded successfully!")
    
    # Verify critical directories exist
    print(f"Input directory: {settings.document_processing.input_directory}")
    print(f"Vector store path: {settings.vector_store.index_path}")
    print(f"API will run on: {settings.api.host}:{settings.api.port}")


if __name__ == "__main__":
    main()