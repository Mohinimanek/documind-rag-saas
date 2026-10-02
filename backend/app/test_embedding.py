from embeddings import generate_embedding

text = "DocuMind is RAG-powered SaaS application."

embedding = generate_embedding(text)

print("Embedding Length:", len(embedding))
print("First 5 values:", embedding[:5])
