#!/usr/bin/env python3
"""
Ingestion script for Portfolio RAG system.

This script:
1. Loads all documents from the knowledge directory
2. Normalizes document format
3. Splits documents into chunks
4. Gets embeddings from Mistral API
5. Creates FAISS index and saves to storage

Usage:
    python scripts/ingest.py

Environment variables required:
    MISTRAL_API_KEY: Your Mistral API key
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import settings
from app.core.splitter import text_splitter
from app.services.mistral_client import mistral_client
from app.services.vector_store import vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DocumentLoader:
    """Handles loading and normalizing documents from various formats."""
    
    def __init__(self, knowledge_dir: str):
        self.knowledge_dir = Path(knowledge_dir)
        
    def load_markdown_file(self, file_path: Path) -> Dict[str, Any]:
        """Load a markdown file and extract metadata."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract title from first # header or use filename
            title = file_path.stem.replace('_', ' ').title()
            lines = content.split('\n')
            
            for line in lines:
                line = line.strip()
                if line.startswith('# '):
                    title = line[2:].strip()
                    break
            
            return {
                'id': file_path.stem,
                'title': title,
                'text': content,
                'source': file_path.name,
                'created_at': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                'file_type': 'markdown'
            }
        except Exception as e:
            logger.error(f"Error loading markdown file {file_path}: {e}")
            return None
    
    def load_json_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load a JSON file containing document data."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, dict):
                # Single document
                documents = [data]
            elif isinstance(data, list):
                # Multiple documents
                documents = data
            else:
                logger.warning(f"Unexpected JSON structure in {file_path}")
                return None
            
            results = []
            for i, doc in enumerate(documents):
                # Normalize document structure
                normalized_doc = {
                    'id': doc.get('id', f"{file_path.stem}_{i}"),
                    'title': doc.get('title', doc.get('name', file_path.stem.replace('_', ' ').title())),
                    'text': doc.get('text', doc.get('content', doc.get('description', ''))),
                    'source': doc.get('source', file_path.name),
                    'created_at': doc.get('created_at', datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()),
                    'file_type': 'json'
                }
                
                # Add technologies field if present
                if 'technologies' in doc:
                    normalized_doc['technologies'] = doc['technologies']
                
                # Add any additional fields
                for key, value in doc.items():
                    if key not in ['id', 'title', 'text', 'source', 'created_at', 'technologies'] and key not in normalized_doc:
                        normalized_doc[key] = value
                
                if normalized_doc['text']:  # Only include docs with text content
                    results.append(normalized_doc)
            
            return results
            
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            return None
    
    def load_all_documents(self) -> List[Dict[str, Any]]:
        """Load all documents from the knowledge directory."""
        if not self.knowledge_dir.exists():
            logger.error(f"Knowledge directory not found: {self.knowledge_dir}")
            return []
        
        documents = []
        
        # Load markdown files
        for md_file in self.knowledge_dir.glob('*.md'):
            logger.info(f"Loading markdown file: {md_file.name}")
            doc = self.load_markdown_file(md_file)
            if doc:
                documents.append(doc)
        
        # Load JSON files  
        for json_file in self.knowledge_dir.glob('*.json'):
            logger.info(f"Loading JSON file: {json_file.name}")
            docs = self.load_json_file(json_file)
            if docs:
                if isinstance(docs, list):
                    documents.extend(docs)
                else:
                    documents.append(docs)
        
        logger.info(f"Loaded {len(documents)} documents total")
        return documents


async def main():
    """Main ingestion pipeline."""
    logger.info("Starting Portfolio RAG ingestion pipeline")
    
    # Validate environment
    if not settings.mistral_api_key:
        logger.error("MISTRAL_API_KEY environment variable not set")
        sys.exit(1)
    
    try:
        # Create storage directory
        storage_dir = Path(settings.faiss_index_path).parent
        storage_dir.mkdir(exist_ok=True)
        
        # Load documents
        loader = DocumentLoader(settings.knowledge_dir)
        documents = loader.load_all_documents()
        
        if not documents:
            logger.error("No documents found to process")
            sys.exit(1)
        
        # Split documents into chunks
        logger.info("Splitting documents into chunks...")
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Extract text for embedding
        texts_to_embed = [chunk['text'] for chunk in chunks]
        
        # Get embeddings from Mistral (in batches)
        logger.info("Getting embeddings from Mistral API...")
        embeddings = []
        batch_size = 100  # Mistral allows up to 128, but we'll be conservative
        
        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts_to_embed) + batch_size - 1)//batch_size} ({len(batch_texts)} texts)")
            
            try:
                batch_embeddings = await mistral_client.embed(batch_texts)
                embeddings.extend(batch_embeddings)
                logger.info(f"Got {len(batch_embeddings)} embeddings")
            except Exception as e:
                logger.error(f"Error getting embeddings for batch: {e}")
                raise
        
        if len(embeddings) != len(chunks):
            raise ValueError(f"Embedding count mismatch: got {len(embeddings)}, expected {len(chunks)}")
        
        # Create and save FAISS index
        logger.info("Creating FAISS index...")
        vector_store.create_index(embeddings, chunks)
        
        logger.info("Saving index and chunks to disk...")
        vector_store.save_index()
        
        # Print summary
        stats = vector_store.get_stats()
        logger.info("Ingestion completed successfully!")
        logger.info(f"Summary:")
        logger.info(f"  - Documents processed: {len(documents)}")
        logger.info(f"  - Chunks created: {len(chunks)}")
        logger.info(f"  - Embeddings generated: {len(embeddings)}")
        logger.info(f"  - Index dimension: {stats['dimension']}")
        logger.info(f"  - Storage files:")
        logger.info(f"    - Index: {settings.faiss_index_path}")
        logger.info(f"    - Chunks: {settings.chunks_path}")
        
        # Test the index with a sample query
        logger.info("Testing index with sample query...")
        test_results = vector_store.search([0.1] * stats['dimension'], k=3)
        logger.info(f"Sample query returned {len(test_results)} results")
        
        logger.info("✅ Ingestion pipeline completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Ingestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Clean up Mistral client
        try:
            await mistral_client.close()
        except:
            pass


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run the async main function
    asyncio.run(main())