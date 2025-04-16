# Semantic Search Service with Qdrant

This is a semantic search service built with FastAPI and Qdrant vector database. It provides endpoints for adding text chunks and performing semantic searches.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Start the Qdrant service using Docker Compose:
```bash
docker-compose up -d
```

3. Run the FastAPI application:
```bash
python main.py
```

The service will be available at `http://localhost:8000`

## API Endpoints

### Add Text Chunk
- **Endpoint**: POST `/add_chunk`
- **Body**:
```json
{
    "text": "Your text chunk here",
    "metadata": {
        "source": "optional metadata",
        "timestamp": "2024-01-01"
    }
}
```

### Search
- **Endpoint**: POST `/search`
- **Body**:
```json
{
    "query": "Your search query",
    "limit": 5
}
```

## Features
- Semantic search using sentence transformers
- Vector storage with Qdrant
- FastAPI for high-performance API endpoints
- Docker support for easy deployment
- Metadata support for text chunks
