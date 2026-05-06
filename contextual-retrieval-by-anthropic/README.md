# Contextual Retrieval by Anthropic

A Python implementation of contextual retrieval inspired by Anthropic's approach to improving document retrieval accuracy.

## Overview

This project implements a contextual retrieval system that enhances document chunks with contextual information before storing them. The key insight is that prepending context to each chunk explaining what the chunk is about in relation to the overall document significantly improves retrieval accuracy.

## Features

- **Document Chunking**: Intelligently splits documents into overlapping chunks with sentence boundary detection
- **Context Generation**: Adds contextual information to each chunk describing its position and content
- **Chunk Storage**: Saves and loads chunks with their context in JSON format
- **Search Functionality**: Simple keyword-based search through contextual chunks
- **Statistics**: Provides insights into stored documents and chunks

## Installation

No external dependencies required - uses only Python standard library.

```bash
cd contextual-retrieval-by-anthropic
```

## Usage

### Basic Example

```python
from src.contextual_retrieval.save_contextual_retrieval import ContextualRetrieval

# Create retrieval system
retrieval = ContextualRetrieval(storage_path="my_storage")

# Add a document
document = """
Your document text here...
"""

chunks = retrieval.add_document(
    document=document,
    doc_id="doc_001",
    metadata={
        'title': 'My Document',
        'author': 'Author Name',
        'date': '2024-01-01'
    },
    chunk_size=500,
    overlap=50
)

# Save chunks to disk
retrieval.save_chunks()

# Search for relevant chunks
results = retrieval.search_chunks("your search query", top_k=5)

# Get statistics
stats = retrieval.get_statistics()
print(stats)
```

### Running the Example

```bash
cd contextual-retrieval-by-anthropic
python src/contextual_retrieval/save_contextual_retrieval.py
```

## How It Works

1. **Chunking**: Documents are split into overlapping chunks with intelligent boundary detection
2. **Context Generation**: For each chunk, a contextual description is generated that includes:
   - Document title (if provided)
   - Position in document (beginning/middle/end)
   - Brief description of chunk content
3. **Storage**: Chunks with their context are stored in JSON format
4. **Retrieval**: When searching, both the context and content are used to find relevant chunks

## Architecture

- `DocumentChunk`: Dataclass representing a chunk with context, content, and metadata
- `ContextualRetrieval`: Main class that handles:
  - Document chunking
  - Context generation
  - Chunk storage and retrieval
  - Search functionality

## Future Enhancements

- **LLM Integration**: Use actual LLMs (like Claude or GPT) for context generation
- **Vector Embeddings**: Implement vector-based similarity search using embeddings
- **BM25 Ranking**: Add BM25 or other advanced ranking algorithms
- **Hybrid Search**: Combine keyword and semantic search
- **Chunk Optimization**: Automatic chunk size optimization based on content
- **Multi-language Support**: Enhanced support for non-English documents

## License

MIT License - See LICENSE file for details

## References

- Anthropic's Contextual Retrieval approach
- RAG (Retrieval Augmented Generation) best practices
