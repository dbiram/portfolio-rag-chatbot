import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import faiss
import numpy as np
from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store for similarity search."""
    
    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Dict[str, Any]] = []
        self.dimension: Optional[int] = None
        self.is_loaded = False
    
    def create_index(self, embeddings: List[List[float]], chunks: List[Dict[str, Any]]) -> None:
        """
        Create a new FAISS index from embeddings and chunks.
        
        Args:
            embeddings: List of embedding vectors
            chunks: List of chunk metadata dictionaries
        """
        if len(embeddings) != len(chunks):
            raise ValueError(f"Embeddings ({len(embeddings)}) and chunks ({len(chunks)}) count mismatch")
        
        if not embeddings:
            raise ValueError("Cannot create index with empty embeddings")
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        self.dimension = embeddings_array.shape[1]
        
        logger.info(f"Creating FAISS index with {len(embeddings)} vectors of dimension {self.dimension}")
        
        # Create FAISS index (using IndexFlatIP for cosine similarity)
        # IndexFlatIP computes inner product, which equals cosine similarity for normalized vectors
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Add vectors to index
        self.index.add(embeddings_array)
        
        # Store chunks
        self.chunks = chunks.copy()
        self.is_loaded = True
        
        logger.info(f"FAISS index created successfully with {self.index.ntotal} vectors")
    
    def save_index(self, index_path: str = None, chunks_path: str = None) -> None:
        """
        Save the FAISS index and chunks to disk.
        
        Args:
            index_path: Path to save FAISS index
            chunks_path: Path to save chunks metadata (JSONL format)
        """
        if not self.is_loaded or self.index is None:
            raise RuntimeError("No index loaded to save")
        
        index_path = index_path or settings.faiss_index_path
        chunks_path = chunks_path or settings.chunks_path
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(chunks_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        logger.info(f"FAISS index saved to {index_path}")
        
        # Save chunks as JSONL
        with open(chunks_path, 'w', encoding='utf-8') as f:
            # First line: metadata about the index
            metadata = {
                "dimension": self.dimension,
                "total_chunks": len(self.chunks),
                "index_type": "IndexFlatIP"
            }
            f.write(json.dumps(metadata) + '\n')
            
            # Subsequent lines: chunks
            for chunk in self.chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        
        logger.info(f"Chunks saved to {chunks_path}")
    
    def load_index(self, index_path: str = None, chunks_path: str = None) -> None:
        """
        Load FAISS index and chunks from disk.
        
        Args:
            index_path: Path to FAISS index file
            chunks_path: Path to chunks metadata file (JSONL format)
        """
        index_path = index_path or settings.faiss_index_path
        chunks_path = chunks_path or settings.chunks_path
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Chunks file not found at {chunks_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        logger.info(f"FAISS index loaded from {index_path}")
        
        # Load chunks from JSONL
        self.chunks = []
        with open(chunks_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            if not lines:
                raise ValueError("Empty chunks file")
            
            # First line is metadata
            metadata = json.loads(lines[0])
            self.dimension = metadata.get("dimension")
            expected_chunks = metadata.get("total_chunks", len(lines) - 1)
            
            # Load chunks
            for line in lines[1:]:
                if line.strip():
                    chunk = json.loads(line)
                    self.chunks.append(chunk)
        
        # Validation
        if len(self.chunks) != expected_chunks:
            logger.warning(f"Expected {expected_chunks} chunks, loaded {len(self.chunks)}")
        
        if self.index.ntotal != len(self.chunks):
            raise ValueError(f"Index size ({self.index.ntotal}) doesn't match chunks count ({len(self.chunks)})")
        
        self.is_loaded = True
        logger.info(f"Loaded {len(self.chunks)} chunks with dimension {self.dimension}")
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar chunks using the query embedding.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            
        Returns:
            List of tuples (chunk_data, similarity_score)
        """
        if not self.is_loaded or self.index is None:
            raise RuntimeError("No index loaded for search")
        
        if len(query_embedding) != self.dimension:
            raise ValueError(f"Query embedding dimension ({len(query_embedding)}) doesn't match index dimension ({self.dimension})")
        
        # Convert to numpy array and normalize
        query_array = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_array)
        
        # Search
        k = min(k, len(self.chunks))  # Don't search for more chunks than we have
        similarities, indices = self.index.search(query_array, k)
        
        # Prepare results
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for empty slots
                break
            
            chunk_data = self.chunks[idx].copy()
            results.append((chunk_data, float(similarity)))
        
        logger.info(f"Found {len(results)} similar chunks for query")
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded index."""
        if not self.is_loaded:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "total_chunks": len(self.chunks),
            "dimension": self.dimension,
            "index_size": self.index.ntotal if self.index else 0
        }


# Global vector store instance
vector_store = VectorStore()