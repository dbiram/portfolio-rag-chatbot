# Portfolio RAG Backend

A minimal FastAPI backend that answers questions about your career using Mistral AI and vector similarity search. Built for easy deployment on Render.

## Features

- 🤖 **RAG-powered Q&A**: Answers questions using your knowledge base with retrieval-augmented generation
- 🔍 **Vector Search**: FAISS-based similarity search for relevant context retrieval
- 🌐 **REST API**: Simple POST `/chat` endpoint with conversation history support
- 🚀 **Render Ready**: Zero-config deployment with included Procfile and render.yaml
- 📝 **Typed & Clean**: Fully typed Python code with Pydantic models
- 🔒 **CORS Enabled**: Ready for frontend integration

## API Endpoints

- `POST /chat` - Chat with Moez's career assistant
- `GET /health` - Health check and system status
- `GET /docs` - Interactive API documentation

## Quick Start

### 1. Local Development

```bash
# Clone and navigate to the project
cd portfolio-rag

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your MISTRAL_API_KEY

# Add your knowledge files to knowledge/ directory
# Supported formats: .md and .json files

# Run ingestion to create vector index
python scripts/ingest.py

# Start the server
python app/main.py
# Or: uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API documentation.

### 2. Deploy to Render

1. **Prepare your repository:**
   ```bash
   # Add your knowledge files to knowledge/
   # Run ingestion locally to create storage/ files
   python scripts/ingest.py
   
   # Commit storage files (required for Render deployment)
   git add storage/
   git commit -m "Add vector index and chunks"
   git push
   ```

2. **Create Render service:**
   - Connect your GitHub repository
   - Use the included `render.yaml` for one-click setup, or:
   - Create a Web Service with:
     - Build Command: `pip install -r requirements.txt`
     - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

3. **Configure environment variables in Render:**
   - `MISTRAL_API_KEY` - Your Mistral API key (required)
   - `FRONTEND_ORIGIN` - Your frontend domain for CORS
   - Other variables are optional (see `.env.example`)

## Configuration

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MISTRAL_API_KEY` | Mistral API key (required) | - |
| `MISTRAL_CHAT_MODEL` | Chat model to use | `mistral-large-latest` |
| `MISTRAL_EMBED_MODEL` | Embedding model | `mistral-embed` |
| `FRONTEND_ORIGIN` | CORS allowed origins | `["http://localhost:5173", "http://localhost:3000"]` |
| `TOP_K` | Number of chunks to retrieve | `5` |

## Knowledge Base Format

### Markdown Files (`knowledge/*.md`)
```markdown
# Document Title

Your content here...
```

### JSON Files (`knowledge/*.json`)
```json
{
  "id": "unique-id",
  "title": "Document Title", 
  "text": "Your content here...",
  "source": "filename.json"
}
```

Or array of documents:
```json
[
  {"title": "Doc 1", "text": "Content 1..."},
  {"title": "Doc 2", "text": "Content 2..."}
]
```

## Architecture

```
portfolio-rag/
├── app/
│   ├── main.py              # FastAPI app & startup
│   ├── api/chat.py          # Chat endpoint
│   ├── core/
│   │   ├── config.py        # Environment configuration
│   │   └── splitter.py      # Text chunking logic
│   ├── models/schemas.py    # Pydantic models
│   └── services/
│       ├── mistral_client.py # Mistral API client
│       ├── vector_store.py   # FAISS vector store
│       ├── retrieval.py      # RAG retrieval logic
│       └── prompt.py         # Prompt engineering
├── knowledge/               # Your .md/.json files
├── storage/                 # Generated index files
├── scripts/ingest.py        # Ingestion pipeline
└── tests/                   # Basic tests
```

## API Usage

### Chat Request
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is your experience with Python?",
    "history": [
      {"role": "user", "content": "Tell me about yourself"},
      {"role": "assistant", "content": "I am a software engineer..."}
    ]
  }'
```

### Response
```json
{
  "answer": "I have extensive Python experience, having worked with it for over 5 years...\n\nSources:\n- Resume (resume.pdf)\n- Projects (projects.md)",
  "sources": [
    {"title": "Resume", "source": "resume.pdf"},
    {"title": "Projects", "source": "projects.md"}
  ]
}
```

## Development

### Running Tests
```bash
pytest tests/
```

### Local Development Server
```bash
# With auto-reload
uvicorn app.main:app --reload --log-level info

# Production-like
python app/main.py
```

### Re-ingesting Content
```bash
# After updating knowledge files
python scripts/ingest.py

# Restart the server to load new index
```

## Troubleshooting

**Vector store not loaded**: Run `python scripts/ingest.py` first
**Empty responses**: Check if knowledge files exist in `knowledge/` directory  
**API errors**: Verify `MISTRAL_API_KEY` is set correctly
**CORS issues**: Update `FRONTEND_ORIGIN` environment variable

## Tech Stack

- **FastAPI** - Modern Python web framework
- **FAISS** - Vector similarity search
- **Mistral AI** - Chat completion and embeddings
- **Pydantic** - Data validation and settings
- **httpx** - Async HTTP client

