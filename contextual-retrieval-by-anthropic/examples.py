"""
Example usage of the Contextual Retrieval System
"""

from src.contextual_retrieval.save_contextual_retrieval import ContextualRetrieval


def example_1_basic_usage():
    """Basic usage example"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Create retrieval system
    retrieval = ContextualRetrieval(storage_path="example_storage")
    
    # Sample document about Python programming
    python_doc = """
    Python Programming Language
    
    Python is a high-level, interpreted programming language known for its simplicity
    and readability. It was created by Guido van Rossum and first released in 1991.
    
    Python supports multiple programming paradigms including procedural, object-oriented,
    and functional programming. Its design philosophy emphasizes code readability with
    significant use of whitespace.
    
    Python has a comprehensive standard library that supports many common programming tasks
    such as file I/O, system calls, and internet protocols. The language also has a large
    ecosystem of third-party packages available through the Python Package Index (PyPI).
    
    Popular applications of Python include web development with frameworks like Django and Flask,
    data science and machine learning with libraries like NumPy, Pandas, and TensorFlow,
    automation and scripting, and scientific computing.
    """
    
    # Add document with metadata
    chunks = retrieval.add_document(
        document=python_doc,
        doc_id="python_intro_001",
        metadata={
            'title': 'Python Programming Language Overview',
            'category': 'Programming',
            'difficulty': 'Beginner'
        },
        chunk_size=300,
        overlap=40
    )
    
    print(f"\nCreated {len(chunks)} chunks")
    
    # Display chunks with their context
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Context: {chunk.context}")
        print(f"Content: {chunk.content[:100]}...")
    
    # Save to disk
    retrieval.save_chunks("python_chunks.json")
    print("\n✓ Chunks saved successfully")


def example_2_search():
    """Search example"""
    print("\n" + "=" * 60)
    print("Example 2: Searching Chunks")
    print("=" * 60)
    
    # Create and populate retrieval system
    retrieval = ContextualRetrieval(storage_path="example_storage")
    
    # Add multiple documents
    docs = [
        {
            'text': """Machine Learning is a subset of artificial intelligence that enables
            systems to learn and improve from experience without being explicitly programmed.
            It focuses on developing computer programs that can access data and use it to learn.""",
            'id': 'ml_001',
            'metadata': {'title': 'Machine Learning Basics', 'topic': 'ML'}
        },
        {
            'text': """Deep Learning is a subset of machine learning based on artificial neural
            networks. The learning can be supervised, semi-supervised or unsupervised. Deep learning
            architectures such as deep neural networks have been applied to fields including computer
            vision, speech recognition, and natural language processing.""",
            'id': 'dl_001',
            'metadata': {'title': 'Deep Learning Introduction', 'topic': 'DL'}
        },
        {
            'text': """Natural Language Processing (NLP) is a branch of artificial intelligence
            that helps computers understand, interpret and manipulate human language. NLP combines
            computational linguistics with statistical, machine learning and deep learning models.""",
            'id': 'nlp_001',
            'metadata': {'title': 'NLP Overview', 'topic': 'NLP'}
        }
    ]
    
    for doc in docs:
        retrieval.add_document(
            document=doc['text'],
            doc_id=doc['id'],
            metadata=doc['metadata'],
            chunk_size=200,
            overlap=30
        )
    
    # Perform searches
    queries = [
        "machine learning",
        "neural networks",
        "human language"
    ]
    
    for query in queries:
        print(f"\nSearch query: '{query}'")
        results = retrieval.search_chunks(query, top_k=2)
        print(f"Found {len(results)} results:")
        for i, chunk in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"  Document: {chunk.metadata.get('title')}")
            print(f"  Content: {chunk.content[:100]}...")


def example_3_statistics():
    """Statistics example"""
    print("\n" + "=" * 60)
    print("Example 3: Getting Statistics")
    print("=" * 60)
    
    retrieval = ContextualRetrieval(storage_path="example_storage")
    
    # Add some documents
    for i in range(3):
        retrieval.add_document(
            document=f"Document {i} content " * 50,
            doc_id=f"doc_{i:03d}",
            metadata={'index': i}
        )
    
    # Get and display statistics
    stats = retrieval.get_statistics()
    print("\nRetrieval System Statistics:")
    print("-" * 40)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")


def example_4_load_and_reuse():
    """Load and reuse saved chunks"""
    print("\n" + "=" * 60)
    print("Example 4: Loading Saved Chunks")
    print("=" * 60)
    
    # Create new retrieval instance
    retrieval = ContextualRetrieval(storage_path="example_storage")
    
    # Try to load previously saved chunks
    print("\nAttempting to load chunks from 'python_chunks.json'...")
    retrieval.load_chunks("python_chunks.json")
    
    if retrieval.chunks:
        print(f"Successfully loaded {len(retrieval.chunks)} chunks")
        
        # Search in loaded chunks
        results = retrieval.search_chunks("python programming", top_k=1)
        if results:
            print(f"\nSearch test successful!")
            print(f"Top result: {results[0].content[:100]}...")
    else:
        print("No chunks loaded - file may not exist")


if __name__ == "__main__":
    # Run all examples
    example_1_basic_usage()
    example_2_search()
    example_3_statistics()
    example_4_load_and_reuse()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
