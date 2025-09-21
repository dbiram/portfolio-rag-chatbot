from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PromptService:
    """Service for building prompts and managing conversation context."""
    
    def __init__(self):
        # System prompt speaking as Moez in first person
        self.system_prompt = """You are Moez's career assistant. Speak as "I" (Moez). Answer using ONLY the provided context about my background, experience, and career.

If something isn't in the context, say you don't have that information available. Be concise and practical in your responses.

If you see truncated or incomplete information (like "[Context truncated...]" or incomplete sentences), do not mention or reference that incomplete information in your answer. Only use complete, clear information from the context.

Do not include a "Sources:" section in your response. Do not cite or mention document names, filenames, or sources within your answer. Just provide a natural, conversational response based on the context.

Do not invent or make up any details not explicitly mentioned in the context but you can paraphrase them to build meaningful paragraphs."""
    
    def build_messages(
        self, 
        question: str, 
        context: str,
        history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """
        Build the message list for the chat completion.
        
        Args:
            question: User's question
            context: Retrieved context from vector store
            history: Optional conversation history
            
        Returns:
            List of message dictionaries for the chat API
        """
        messages = []
        
        # System message
        messages.append({
            "role": "system",
            "content": self.system_prompt
        })
        
        # Add conversation history if provided (but limit it)
        if history:
            # Limit history to last 6 messages to control token usage
            limited_history = self._limit_history(history)
            messages.extend(limited_history)
        
        # User message with context and question
        user_content = self._build_user_message(question, context)
        messages.append({
            "role": "user", 
            "content": user_content
        })
        
        return messages
    
    def _limit_history(self, history: List[Dict[str, str]], max_messages: int = 6) -> List[Dict[str, str]]:
        """
        Limit conversation history to prevent token overflow.
        
        Args:
            history: Full conversation history
            max_messages: Maximum number of messages to include
            
        Returns:
            Trimmed history
        """
        if len(history) <= max_messages:
            return history
        
        # Take the most recent messages
        limited = history[-max_messages:]
        
        logger.info(f"Limited history from {len(history)} to {len(limited)} messages")
        return limited
    
    def _build_user_message(self, question: str, context: str) -> str:
        """
        Build the user message combining context and question.
        
        Args:
            question: User's question
            context: Retrieved context
            
        Returns:
            Formatted user message
        """
        # Truncate context if it's very long to prevent token overflow
        max_context_length = 3000  # Increased limit since we have smaller chunks now
        if len(context) > max_context_length:
            # Try to truncate at sentence boundary
            truncated = context[:max_context_length]
            last_period = truncated.rfind('.')
            if last_period > max_context_length * 0.8:  # If we found a period reasonably close to end
                context = truncated[:last_period + 1] + "\n\n[Content continues but truncated for brevity]"
            else:
                context = truncated + "\n\n[Content continues but truncated for brevity]"
            
            logger.warning(f"Context truncated to {len(context)} characters")
        
        user_message = f"""{context}

**Question:** {question}

IMPORTANT: Answer as Moez speaking in first person. DO NOT include any "Sources:" section, DO NOT mention document names, filenames, or citations. Just provide a natural conversational response based on the context above."""
        
        return user_message
    
    def extract_question_intent(self, question: str) -> Dict[str, Any]:
        """
        Extract basic intent and metadata from the question.
        This could be extended for more sophisticated query processing.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with intent information
        """
        question_lower = question.lower().strip()
        
        # Basic intent classification
        intent_keywords = {
            "experience": ["experience", "worked", "job", "role", "position", "career"],
            "skills": ["skills", "technologies", "tools", "programming", "languages"],
            "education": ["education", "degree", "university", "college", "studied", "learning"],
            "projects": ["projects", "built", "created", "developed", "portfolio"],
            "contact": ["contact", "email", "phone", "reach", "linkedin"],
            "general": []  # fallback
        }
        
        detected_intent = "general"
        confidence = 0.0
        
        for intent, keywords in intent_keywords.items():
            if intent == "general":
                continue
            
            matches = sum(1 for keyword in keywords if keyword in question_lower)
            if matches > 0:
                current_confidence = matches / len(keywords)
                if current_confidence > confidence:
                    detected_intent = intent
                    confidence = current_confidence
        
        return {
            "intent": detected_intent,
            "confidence": confidence,
            "length": len(question),
            "word_count": len(question.split())
        }
    
    def validate_question(self, question: str) -> Dict[str, Any]:
        """
        Validate the user's question.
        
        Args:
            question: User's question
            
        Returns:
            Validation result with is_valid and error message
        """
        result = {"is_valid": True, "error": None}
        
        if not question:
            result["is_valid"] = False
            result["error"] = "Question cannot be empty"
            return result
        
        question = question.strip()
        if not question:
            result["is_valid"] = False
            result["error"] = "Question cannot be empty"
            return result
        
        if len(question) > 1000:  # Reasonable limit
            result["is_valid"] = False
            result["error"] = "Question too long (max 1000 characters)"
            return result
        
        if len(question) < 3:
            result["is_valid"] = False
            result["error"] = "Question too short"
            return result
        
        return result
    
    def format_sources_in_response(self, response: str, sources: List[Dict[str, str]]) -> str:
        """
        Ensure the response includes properly formatted sources.
        This is a fallback in case the LLM doesn't format sources correctly.
        
        Args:
            response: LLM response
            sources: List of source dictionaries
            
        Returns:
            Response with properly formatted sources
        """
        if not sources:
            if "sources:" not in response.lower():
                response += "\n\nSources: No specific sources available."
            return response
        
        # Check if response already has sources section
        if "sources:" in response.lower():
            return response
        
        # Add sources section
        sources_text = "\n\nSources:\n"
        for i, source in enumerate(sources, 1):
            title = source.get('title', 'Untitled')
            source_location = source.get('source', 'Unknown')
            sources_text += f"- {title} ({source_location})\n"
        
        return response + sources_text


# Global prompt service instance  
prompt_service = PromptService()