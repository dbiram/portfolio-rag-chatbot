import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert data["status"] in ["ok", "degraded", "error"]
    assert "timestamp" in data
    assert "vector_store" in data


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "message" in data
    assert "docs" in data
    assert "health" in data


def test_docs_accessible():
    """Test that the API documentation is accessible."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_404_handler():
    """Test the custom 404 handler."""
    response = client.get("/nonexistent")
    
    assert response.status_code == 404
    data = response.json()
    
    assert "error" in data
    assert data["error"] == "Not Found"
    assert "available_endpoints" in data