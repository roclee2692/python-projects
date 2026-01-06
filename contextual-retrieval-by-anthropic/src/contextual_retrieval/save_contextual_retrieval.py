"""
Contextual Retrieval System - Save and Store Module

This module implements a contextual retrieval system that enhances document chunks
with contextual information before storing them for retrieval.

The key idea is to prepend context to each chunk explaining what the chunk is about
in relation to the overall document, improving retrieval accuracy.
"""

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import hashlib

# Constants
MIN_CHUNK_RATIO = 0.5  # Minimum ratio of chunk size to break at sentence boundary


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with its context"""
    chunk_id: str
    content: str
    context: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentChunk':
        """Create chunk from dictionary"""
        return cls(**data)


class ContextualRetrieval:
    """
    Contextual Retrieval System
    
    This system breaks documents into chunks and adds contextual information
    to each chunk to improve retrieval accuracy.
    """
    
    def __init__(self, storage_path: str = "contextual_storage"):
        """
        Initialize the contextual retrieval system
        
        Args:
            storage_path: Directory path to store chunks and metadata
        """
        self.storage_path = storage_path
        self.chunks: List[DocumentChunk] = []
        os.makedirs(storage_path, exist_ok=True)
    
    def _generate_chunk_id(self, content: str, doc_id: str, index: int) -> str:
        """
        Generate a unique ID for a chunk
        
        Args:
            content: The chunk content
            doc_id: The document ID
            index: The chunk index in the document
            
        Returns:
            A unique chunk ID
        """
        hash_input = f"{doc_id}_{index}_{content[:100]}"
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def _chunk_document(self, document: str, chunk_size: int = 500, 
                       overlap: int = 50) -> List[str]:
        """
        Split document into overlapping chunks
        
        Args:
            document: The document text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Number of overlapping characters between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        doc_length = len(document)
        
        while start < doc_length:
            end = start + chunk_size
            chunk = document[start:end]
            
            # Try to break at sentence boundary if possible
            if end < doc_length:
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                # Only break if we found a boundary and we're past the minimum ratio
                if break_point > chunk_size * MIN_CHUNK_RATIO:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            # Only append non-empty chunks
            if chunk.strip():
                chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks
    
    def _generate_context(self, chunk: str, document: str, 
                         document_title: str = "", chunk_index: int = 0,
                         total_chunks: int = 1) -> str:
        """
        Generate contextual information for a chunk
        
        In a real implementation, this would use an LLM to generate context.
        For this basic implementation, we'll create a simple context string.
        
        Args:
            chunk: The text chunk
            document: The full document
            document_title: Optional title of the document
            chunk_index: Index of this chunk in the document
            total_chunks: Total number of chunks in the document
            
        Returns:
            Contextual description of the chunk
        """
        # Simple context generation (in production, use an LLM)
        context = f"This chunk is from "
        if document_title:
            context += f"a document titled '{document_title}'. "
        else:
            context += "a document. "
        
        # Determine position based on chunk index rather than text search
        if total_chunks > 1:
            position_ratio = chunk_index / (total_chunks - 1) if total_chunks > 1 else 0
            if position_ratio < 0.3:
                context += "This is from the beginning of the document. "
            elif position_ratio > 0.7:
                context += "This is from the end of the document. "
            else:
                context += "This is from the middle of the document. "
        else:
            context += "This is the complete document. "
        
        # Add a snippet of what the chunk discusses
        first_sentence = chunk.split('.')[0] if '.' in chunk else chunk[:100]
        context += f"It discusses: {first_sentence.strip()}..."
        
        return context
    
    def add_document(self, document: str, doc_id: str, 
                    metadata: Optional[Dict[str, Any]] = None,
                    chunk_size: int = 500, overlap: int = 50) -> List[DocumentChunk]:
        """
        Add a document to the retrieval system with contextual chunks
        
        Args:
            document: The document text
            doc_id: Unique identifier for the document
            metadata: Optional metadata about the document
            chunk_size: Maximum characters per chunk
            overlap: Number of overlapping characters between chunks
            
        Returns:
            List of created DocumentChunk objects
        """
        if metadata is None:
            metadata = {}
        
        # Chunk the document
        chunks = self._chunk_document(document, chunk_size, overlap)
        
        document_title = metadata.get('title', '')
        created_chunks = []
        
        # Create contextual chunks
        for i, chunk_text in enumerate(chunks):
            # Generate context for this chunk
            context = self._generate_context(
                chunk_text, document, document_title, 
                chunk_index=i, total_chunks=len(chunks)
            )
            
            # Create chunk object
            chunk_id = self._generate_chunk_id(chunk_text, doc_id, i)
            chunk_metadata = {
                **metadata,
                'doc_id': doc_id,
                'chunk_index': i,
                'total_chunks': len(chunks)
            }
            
            doc_chunk = DocumentChunk(
                chunk_id=chunk_id,
                content=chunk_text,
                context=context,
                metadata=chunk_metadata
            )
            
            created_chunks.append(doc_chunk)
            self.chunks.append(doc_chunk)
        
        return created_chunks
    
    def save_chunks(self, filename: str = "chunks.json") -> None:
        """
        Save all chunks to a JSON file
        
        Args:
            filename: Name of the file to save to
        """
        filepath = os.path.join(self.storage_path, filename)
        
        chunks_data = [chunk.to_dict() for chunk in self.chunks]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(self.chunks)} chunks to {filepath}")
    
    def load_chunks(self, filename: str = "chunks.json") -> None:
        """
        Load chunks from a JSON file
        
        Args:
            filename: Name of the file to load from
        """
        filepath = os.path.join(self.storage_path, filename)
        
        if not os.path.exists(filepath):
            print(f"No file found at {filepath}")
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        self.chunks = [DocumentChunk.from_dict(data) for data in chunks_data]
        print(f"Loaded {len(self.chunks)} chunks from {filepath}")
    
    def get_contextual_content(self, chunk: DocumentChunk) -> str:
        """
        Get the full contextual content (context + content) for a chunk
        
        Args:
            chunk: The document chunk
            
        Returns:
            Combined context and content string
        """
        return f"{chunk.context}\n\n{chunk.content}"
    
    def search_chunks(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """
        Simple keyword-based search through chunks
        
        In production, this would use vector similarity search with embeddings.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of matching chunks
        """
        query_lower = query.lower()
        
        # Score chunks by keyword matches in context + content
        scored_chunks = []
        for chunk in self.chunks:
            contextual_content = self.get_contextual_content(chunk).lower()
            score = sum(1 for word in query_lower.split() 
                       if word in contextual_content)
            if score > 0:
                scored_chunks.append((score, chunk))
        
        # Sort by score and return top_k
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, chunk in scored_chunks[:top_k]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the stored chunks
        
        Returns:
            Dictionary with statistics
        """
        if not self.chunks:
            return {
                'total_chunks': 0,
                'total_documents': 0
            }
        
        doc_ids = set(chunk.metadata.get('doc_id') for chunk in self.chunks)
        avg_chunk_length = sum(len(chunk.content) for chunk in self.chunks) / len(self.chunks)
        avg_context_length = sum(len(chunk.context) for chunk in self.chunks) / len(self.chunks)
        
        return {
            'total_chunks': len(self.chunks),
            'total_documents': len(doc_ids),
            'average_chunk_length': avg_chunk_length,
            'average_context_length': avg_context_length,
            'storage_path': self.storage_path
        }


def main():
    """Example usage of the contextual retrieval system"""
    # Create retrieval system
    retrieval = ContextualRetrieval()
    
    # Example document
    sample_doc = """
    Artificial Intelligence and Machine Learning
    
    Artificial intelligence (AI) is the simulation of human intelligence by machines.
    Machine learning is a subset of AI that enables systems to learn from data.
    Deep learning is a subset of machine learning that uses neural networks.
    
    Natural Language Processing (NLP) is a branch of AI that helps computers understand
    and process human language. NLP is used in chatbots, translation, and text analysis.
    
    Computer Vision is another important AI field that enables machines to interpret
    visual information from the world. It's used in facial recognition, autonomous vehicles,
    and medical image analysis.
    
    The future of AI includes advancements in general AI, ethical AI, and AI safety.
    Researchers are working on making AI more transparent, fair, and beneficial to humanity.
    """
    
    # Add document
    chunks = retrieval.add_document(
        document=sample_doc,
        doc_id="ai_overview_001",
        metadata={
            'title': 'Introduction to AI and Machine Learning',
            'author': 'AI Research Team',
            'date': '2024-01-01'
        },
        chunk_size=200,
        overlap=30
    )
    
    print(f"Created {len(chunks)} chunks\n")
    
    # Display first chunk with context
    print("Example Chunk with Context:")
    print("-" * 50)
    first_chunk = chunks[0]
    print(f"Chunk ID: {first_chunk.chunk_id}")
    print(f"\nContext: {first_chunk.context}")
    print(f"\nContent: {first_chunk.content}")
    print("-" * 50)
    
    # Save chunks
    retrieval.save_chunks()
    
    # Show statistics
    print("\nStatistics:")
    stats = retrieval.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Search example
    print("\nSearch Example:")
    results = retrieval.search_chunks("natural language processing")
    print(f"Found {len(results)} results for 'natural language processing'")
    if results:
        print(f"\nTop result context: {results[0].context}")


if __name__ == "__main__":
    main()
