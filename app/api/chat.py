import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status
from app.models.schemas import ChatRequest, ChatResponse, Source, ErrorResponse
from app.services.retrieval import retrieval_service
from app.services.prompt import prompt_service
from app.services.mistral_client import mistral_client, MistralAPIError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    },
    summary="Chat with Moez's career assistant",
    description="Ask questions about Moez's background, experience, and career using RAG-powered responses"
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Process a chat request and return an answer based on Moez's knowledge base.
    
    The endpoint:
    1. Validates the question
    2. Retrieves relevant context from the vector store
    3. Builds a prompt with context and conversation history
    4. Calls Mistral API for chat completion
    5. Returns the answer with sources
    """
    start_time = datetime.now()
    
    try:
        # Validate the question
        validation = prompt_service.validate_question(request.question)
        if not validation["is_valid"]:
            logger.warning(f"Invalid question: {validation['error']}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=validation["error"]
            )
        
        logger.info(f"Processing chat request: '{request.question[:100]}...'")
        
        # Extract question intent (for logging/analytics)
        intent_info = prompt_service.extract_question_intent(request.question)
        logger.info(f"Detected intent: {intent_info['intent']} (confidence: {intent_info['confidence']:.2f})")
        
        # Retrieve relevant context and sources
        try:
            context, sources = await retrieval_service.retrieve_and_format(request.question)
        except RuntimeError as e:
            logger.error(f"Retrieval failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Knowledge base not available. Please contact support."
            )
        except Exception as e:
            logger.error(f"Unexpected retrieval error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve relevant information"
            )
        
        # Convert history to the format expected by prompt service
        history = None
        if request.history:
            history = [{"role": msg.role, "content": msg.content} for msg in request.history]
        
        # Build messages for chat completion
        messages = prompt_service.build_messages(
            question=request.question,
            context=context,
            history=history
        )
        
        # Call Mistral chat API
        try:
            answer = await mistral_client.chat(messages)
        except MistralAPIError as e:
            logger.error(f"Mistral API error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate response. Please try again."
            )
        except Exception as e:
            logger.error(f"Unexpected chat error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate response"
            )
        
        # Use the raw answer without adding sources to the text
        formatted_answer = answer
        
        # Convert sources to response format
        response_sources = [
            Source(
                title=source["title"],
                source=source["source"],
                created_at=source.get("created_at")
            )
            for source in sources
        ]
        
        # Log successful completion
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Chat request completed in {duration:.2f}s with {len(response_sources)} sources")
        
        return ChatResponse(
            answer=formatted_answer,
            sources=response_sources
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log unexpected errors and return generic error response
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Unexpected error in chat endpoint after {duration:.2f}s: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )


@router.get(
    "/chat/stats",
    summary="Get chat system statistics",
    description="Returns statistics about the retrieval system and knowledge base"
)
async def get_chat_stats() -> Dict[str, Any]:
    """Get statistics about the chat system."""
    try:
        stats = retrieval_service.get_retrieval_stats()
        stats["timestamp"] = datetime.now().isoformat()
        return stats
    except Exception as e:
        logger.error(f"Error getting chat stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )