import logging
from typing import List, Dict, Any, Tuple
from app.services.mistral_client import mistral_client
from app.services.vector_store import vector_store
from app.core.config import settings

logger = logging.getLogger(__name__)


class RetrievalService:
    """Service for retrieving relevant chunks based on query similarity."""
    
    def __init__(self):
        self.vector_store = vector_store
        self.mistral_client = mistral_client
    
    async def retrieve_chunks(self, query: str, top_k: int = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve the most relevant chunks for a given query.
        
        Args:
            query: User query string
            top_k: Number of chunks to retrieve (defaults to settings.top_k)
            
        Returns:
            List of tuples (chunk_data, similarity_score) ordered by relevance
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        top_k = top_k or settings.top_k
        
        # Check if vector store is loaded
        if not self.vector_store.is_loaded:
            raise RuntimeError("Vector store not loaded. Please run ingestion first.")
        
        logger.info(f"Retrieving top {top_k} chunks for query: '{query[:100]}...'")
        
        try:
            # Get query embedding
            query_embeddings = await self.mistral_client.embed([query])
            query_embedding = query_embeddings[0]
            
            # Search for similar chunks (get more than needed to allow for re-ranking)
            search_k = min(top_k * 3, self.vector_store.index.ntotal)
            results = self.vector_store.search(query_embedding, k=search_k)
            
            # Apply source-based boosting
            boosted_results = []
            for chunk_data, similarity_score in results:
                source = chunk_data.get('source', '')
                boost_factor = settings.source_priority_boost.get(source, 1.0)
                boosted_score = similarity_score * boost_factor
                boosted_results.append((chunk_data, boosted_score))
                
                logger.debug(f"Chunk '{chunk_data.get('title', 'Untitled')}' from {source}: "
                           f"original={similarity_score:.3f}, boosted={boosted_score:.3f} (boost={boost_factor})")
            
            # Re-sort by boosted scores and take top_k
            boosted_results.sort(key=lambda x: x[1], reverse=True)
            final_results = boosted_results[:top_k]
            
            logger.info(f"Retrieved {len(final_results)} chunks with boosted similarities: "
                       f"{[f'{score:.3f}' for _, score in final_results]}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error during chunk retrieval: {e}")
            raise RuntimeError(f"Failed to retrieve chunks: {e}")
    
    def format_chunks_for_context(self, chunks_with_scores: List[Tuple[Dict[str, Any], float]]) -> str:
        """
        Format retrieved chunks into a context string for the LLM.
        
        Args:
            chunks_with_scores: List of (chunk_data, similarity_score) tuples
            
        Returns:
            Formatted context string
        """
        if not chunks_with_scores:
            return "No relevant information found."
        
        context_parts = []
        
        for i, (chunk_data, score) in enumerate(chunks_with_scores, 1):
            # Extract key information
            text = chunk_data.get('text', '').strip()
            title = chunk_data.get('title', 'Untitled')
            source = chunk_data.get('source', 'Unknown')
            
            # Truncate text if too long (keep context manageable)
            max_chunk_length = 500
            if len(text) > max_chunk_length:
                text = text[:max_chunk_length] + "..."
            
            # Format the chunk
            context_parts.append(f"{i}. **{title}** (from {source})\n   {text}")
        
        return "**Context:**\n" + "\n\n".join(context_parts)
    
    def extract_sources(self, chunks_with_scores: List[Tuple[Dict[str, Any], float]]) -> List[Dict[str, str]]:
        """
        Extract unique sources from retrieved chunks.
        
        Args:
            chunks_with_scores: List of (chunk_data, similarity_score) tuples
            
        Returns:
            List of source dictionaries with title and source
        """
        seen_sources = set()
        sources = []
        
        for chunk_data, _ in chunks_with_scores:
            title = chunk_data.get('title', 'Untitled')
            source = chunk_data.get('source', 'Unknown')
            doc_id = chunk_data.get('id', f"{title}_{source}")
            
            # Use doc_id to deduplicate sources from same document
            if doc_id not in seen_sources:
                seen_sources.add(doc_id)
                
                source_info = {"title": title, "source": source}
                
                # Add optional fields if they exist
                if 'created_at' in chunk_data:
                    source_info['created_at'] = chunk_data['created_at']
                
                sources.append(source_info)
        
        return sources
    
    async def retrieve_and_format(self, query: str, top_k: int = None) -> Tuple[str, List[Dict[str, str]]]:
        """
        Retrieve chunks and return both formatted context and sources.
        
        Args:
            query: User query string
            top_k: Number of chunks to retrieve
            
        Returns:
            Tuple of (formatted_context, sources_list)
        """
        try:
            # Retrieve chunks
            chunks_with_scores = await self.retrieve_chunks(query, top_k)
            
            # Format context and extract sources
            context = self.format_chunks_for_context(chunks_with_scores)
            sources = self.extract_sources(chunks_with_scores)
            
            return context, sources
            
        except Exception as e:
            logger.error(f"Error in retrieve_and_format: {e}")
            # Return fallback response
            return "I don't have information about that topic in my knowledge base.", []
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval system."""
        vector_stats = self.vector_store.get_stats()
        
        return {
            "vector_store": vector_stats,
            "top_k": settings.top_k,
            "embed_model": settings.mistral_embed_model
        }


# Global retrieval service instance
retrieval_service = RetrievalService()