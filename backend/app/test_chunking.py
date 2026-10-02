from chunking import chunk_text

text = """
DocuMind is a RAG-powered SaaS application.
It allows users to upload document.
The documents are converted into text.
The text is divided into smaller chunks.
These chunks will later be converted into embeddings.
The embeddings will be used for semantic search.
"""

chunks = chunk_text(text, chunk_size=100, overlap=20)

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk{i+1}---")
    print(chunk)