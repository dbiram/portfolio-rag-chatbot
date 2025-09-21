import asyncio
import logging
from typing import List, Dict, Any
import httpx
from app.core.config import settings

logger = logging.getLogger(__name__)


class MistralAPIError(Exception):
    """Custom exception for Mistral API errors."""
    pass


class MistralClient:
    def __init__(self):
        self.api_key = settings.mistral_api_key
        self.base_url = settings.mistral_base_url
        self.chat_model = settings.mistral_chat_model
        self.embed_model = settings.mistral_embed_model
        
        # HTTP client with timeout and retry configuration
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0),
            headers={
                "Authorization": f"Bearer {self._mask_api_key()}",
                "Content-Type": "application/json",
                "User-Agent": "portfolio-rag/1.0"
            }
        )
    
    def _mask_api_key(self) -> str:
        """Return masked API key for logging purposes."""
        if not self.api_key:
            return "MISSING"
        return f"{self.api_key[:8]}...{self.api_key[-4:]}" if len(self.api_key) > 12 else "***"
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers with the actual API key."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "portfolio-rag/1.0"
        }
    
    async def _make_request(self, method: str, endpoint: str, data: Dict[str, Any], retries: int = 3) -> Dict[str, Any]:
        """Make HTTP request with exponential backoff retry."""
        url = f"{self.base_url.rstrip('/')}{endpoint}"
        
        for attempt in range(retries):
            try:
                response = await self.client.request(
                    method=method,
                    url=url,
                    json=data,
                    headers=self._get_headers()
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limit
                    if attempt < retries - 1:
                        wait_time = (2 ** attempt) + 1  # Exponential backoff: 2, 5, 9 seconds
                        logger.warning(f"Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{retries}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise MistralAPIError(f"Rate limited after {retries} attempts")
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    logger.error(f"Mistral API error: {error_msg}")
                    raise MistralAPIError(error_msg)
                    
            except httpx.TimeoutException:
                if attempt < retries - 1:
                    wait_time = (2 ** attempt) + 1
                    logger.warning(f"Request timeout. Retrying in {wait_time}s (attempt {attempt + 1}/{retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise MistralAPIError(f"Request timeout after {retries} attempts")
            except httpx.RequestError as e:
                if attempt < retries - 1:
                    wait_time = (2 ** attempt) + 1
                    logger.warning(f"Request error: {e}. Retrying in {wait_time}s (attempt {attempt + 1}/{retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise MistralAPIError(f"Request error after {retries} attempts: {e}")
        
        raise MistralAPIError(f"Failed after {retries} attempts")
    
    async def chat(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """
        Send chat completion request to Mistral API.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        if not messages:
            raise ValueError("Messages cannot be empty")
        
        # Use settings defaults if not provided
        temperature = temperature if temperature is not None else settings.temperature
        max_tokens = max_tokens if max_tokens is not None else settings.max_tokens
        
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        logger.info(f"Sending chat request with {len(messages)} messages to model {self.chat_model}")
        
        try:
            response_data = await self._make_request("POST", "/v1/chat/completions", payload)
            
            # Extract the response text
            choices = response_data.get("choices", [])
            if not choices:
                raise MistralAPIError("No choices in response")
            
            message = choices[0].get("message", {})
            content = message.get("content", "")
            
            if not content:
                logger.warning("Empty content in chat response")
                return "I apologize, but I couldn't generate a response at this time."
            
            logger.info("Chat request completed successfully")
            return content.strip()
            
        except Exception as e:
            logger.error(f"Chat request failed: {e}")
            raise MistralAPIError(f"Chat completion failed: {e}")
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed (max 128 per batch)
            
        Returns:
            List of embedding vectors (one per input text)
        """
        if not texts:
            raise ValueError("Texts cannot be empty")
        
        if len(texts) > 128:
            raise ValueError("Maximum 128 texts per batch")
        
        # Filter out empty texts
        non_empty_texts = [text for text in texts if text.strip()]
        if len(non_empty_texts) != len(texts):
            logger.warning(f"Filtered out {len(texts) - len(non_empty_texts)} empty texts")
        
        if not non_empty_texts:
            raise ValueError("No non-empty texts to embed")
        
        payload = {
            "model": self.embed_model,
            "input": non_empty_texts,
            "encoding_format": "float"
        }
        
        logger.info(f"Sending embedding request for {len(non_empty_texts)} texts to model {self.embed_model}")
        
        try:
            response_data = await self._make_request("POST", "/v1/embeddings", payload)
            
            # Extract embeddings
            data = response_data.get("data", [])
            if not data:
                raise MistralAPIError("No embeddings in response")
            
            # Sort by index to maintain order
            data.sort(key=lambda x: x.get("index", 0))
            
            embeddings = [item["embedding"] for item in data]
            
            if len(embeddings) != len(non_empty_texts):
                raise MistralAPIError(f"Expected {len(non_empty_texts)} embeddings, got {len(embeddings)}")
            
            logger.info(f"Embedding request completed successfully, got {len(embeddings)} vectors")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding request failed: {e}")
            raise MistralAPIError(f"Embedding failed: {e}")
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Global client instance
mistral_client = MistralClient()