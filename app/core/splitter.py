import re
from typing import List, Dict, Any
from app.core.config import settings


class TextSplitter:
    """
    Lightweight text splitter that chunks text into approximately target_size tokens
    with overlap, using sentence boundaries when possible.
    """
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        # Sentence boundary patterns (ordered by priority)
        self.sentence_endings = [
            r'\.\s+',     # Period followed by whitespace
            r'!\s+',      # Exclamation followed by whitespace  
            r'\?\s+',     # Question mark followed by whitespace
            r'\.\n',      # Period followed by newline
            r'!\n',       # Exclamation followed by newline
            r'\?\n',      # Question mark followed by newline
        ]
        
        # Paragraph/section boundaries
        self.section_endings = [
            r'\n\n+',     # Double newline (paragraph break)
            r'\n#{1,6}\s', # Markdown headers
            r'\n\*\s',    # Markdown list items
            r'\n\d+\.\s', # Numbered lists
            r'\n-\s',     # Dashed lists
        ]
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Rough token estimation: ~4 characters per token for English text.
        This is a simple heuristic - actual tokenization would be more accurate.
        """
        return max(1, len(text) // 4)
    
    def _find_best_split_point(self, text: str, max_pos: int) -> int:
        """
        Find the best position to split text before max_pos, prioritizing sentence boundaries.
        """
        if max_pos >= len(text):
            return len(text)
        
        # Try sentence endings first
        for pattern in self.sentence_endings:
            matches = list(re.finditer(pattern, text[:max_pos]))
            if matches:
                # Get the last match (closest to max_pos)
                last_match = matches[-1]
                split_pos = last_match.end()
                if split_pos > max_pos * 0.5:  # Don't split too early
                    return split_pos
        
        # Try section endings
        for pattern in self.section_endings:
            matches = list(re.finditer(pattern, text[:max_pos]))
            if matches:
                last_match = matches[-1]
                split_pos = last_match.start() + 1  # Keep the newline with previous chunk
                if split_pos > max_pos * 0.3:
                    return split_pos
        
        # Fallback to word boundary
        word_boundary = text.rfind(' ', 0, max_pos)
        if word_boundary > max_pos * 0.3:
            return word_boundary + 1
        
        # Final fallback to character split
        return max_pos
    
    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Text to split
            metadata: Base metadata to include with each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            return []
        
        text = text.strip()
        chunks = []
        
        # If text is smaller than chunk size, return as single chunk
        if self._estimate_tokens(text) <= self.chunk_size:
            chunk_data = {
                "text": text,
                "chunk_index": 0,
                "total_chunks": 1,
                "tokens": self._estimate_tokens(text)
            }
            if metadata:
                chunk_data.update(metadata)
            return [chunk_data]
        
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Estimate where this chunk should end
            estimated_end_chars = start + (self.chunk_size * 4)  # ~4 chars per token
            
            if estimated_end_chars >= len(text):
                # Last chunk
                chunk_text = text[start:].strip()
                if chunk_text:
                    chunk_data = {
                        "text": chunk_text,
                        "chunk_index": chunk_index,
                        "tokens": self._estimate_tokens(chunk_text)
                    }
                    if metadata:
                        chunk_data.update(metadata)
                    chunks.append(chunk_data)
                break
            
            # Find the best split point
            split_point = self._find_best_split_point(text, estimated_end_chars)
            chunk_text = text[start:split_point].strip()
            
            if chunk_text:
                chunk_data = {
                    "text": chunk_text,
                    "chunk_index": chunk_index,
                    "tokens": self._estimate_tokens(chunk_text)
                }
                if metadata:
                    chunk_data.update(metadata)
                chunks.append(chunk_data)
                chunk_index += 1
            
            # Calculate next start position with overlap
            if chunk_index > 0:
                # Find overlap start point
                overlap_chars = self.chunk_overlap * 4  # ~4 chars per token
                overlap_start = max(start, split_point - overlap_chars)
                
                # Try to start overlap at a sentence boundary
                overlap_text = text[overlap_start:split_point]
                sentence_start = None
                
                for pattern in self.sentence_endings:
                    matches = list(re.finditer(pattern, overlap_text))
                    if matches:
                        # Start from the last sentence in overlap
                        sentence_start = overlap_start + matches[-1].end()
                        break
                
                start = sentence_start if sentence_start else max(overlap_start, start + 1)
            else:
                start = split_point
        
        # Update total_chunks count
        for chunk in chunks:
            chunk["total_chunks"] = len(chunks)
        
        return chunks
    
    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split multiple documents into chunks.
        
        Args:
            documents: List of document dicts with 'text' field and optional metadata
            
        Returns:
            List of chunk dictionaries
        """
        all_chunks = []
        
        for doc in documents:
            text = doc.get('text', '')
            if not text:
                continue
            
            # Extract metadata (exclude 'text' field)
            metadata = {k: v for k, v in doc.items() if k != 'text'}
            
            # Split the document
            doc_chunks = self.split_text(text, metadata)
            
            # Add global chunk IDs
            for i, chunk in enumerate(doc_chunks):
                chunk['global_chunk_id'] = len(all_chunks) + i
            
            all_chunks.extend(doc_chunks)
        
        return all_chunks


# Default splitter instance
text_splitter = TextSplitter()