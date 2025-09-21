from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Chat message model for conversation history."""
    role: str = Field(..., description="Role of the message sender", pattern="^(user|assistant)$")
    content: str = Field(..., description="Content of the message")


class ChatRequest(BaseModel):
    """Request model for the /chat endpoint."""
    question: str = Field(..., description="User's question", min_length=1, max_length=1000)
    history: Optional[List[ChatMessage]] = Field(default=None, description="Conversation history")
    
    class Config:
        schema_extra = {
            "example": {
                "question": "What is your experience with Python development?",
                "history": [
                    {"role": "user", "content": "Tell me about your background"},
                    {"role": "assistant", "content": "I'm a software engineer with 5 years of experience..."}
                ]
            }
        }


class Source(BaseModel):
    """Source information for retrieved documents."""
    title: str = Field(..., description="Title of the source document")
    source: str = Field(..., description="Source location or filename")
    created_at: Optional[str] = Field(default=None, description="Document creation date")


class ChatResponse(BaseModel):
    """Response model for the /chat endpoint."""
    answer: str = Field(..., description="Generated answer from Moez's perspective")
    sources: List[Source] = Field(default=[], description="Sources used to generate the answer")
    
    class Config:
        schema_extra = {
            "example": {
                "answer": "I have extensive experience with Python development, having worked with it for over 5 years. I've built web applications using frameworks like Django and FastAPI...\n\nSources:\n- Resume (resume.pdf)\n- Project Portfolio (projects.md)",
                "sources": [
                    {"title": "Resume", "source": "resume.pdf"},
                    {"title": "Project Portfolio", "source": "projects.md"}
                ]
            }
        }


class HealthResponse(BaseModel):
    """Response model for the /health endpoint."""
    status: str = Field(..., description="Service status")
    timestamp: Optional[str] = Field(default=None, description="Response timestamp")
    vector_store: Optional[Dict[str, Any]] = Field(default=None, description="Vector store status")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "ok",
                "timestamp": "2024-01-15T10:30:00Z",
                "vector_store": {
                    "loaded": True,
                    "total_chunks": 150,
                    "dimension": 1024
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Invalid request",
                "detail": "Question cannot be empty"
            }
        }