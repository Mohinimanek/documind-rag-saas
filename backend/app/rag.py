from sqlalchemy.orm import Session 
from .search import search_chunks

def retrieve_context(
    query: str,
    db: Session,
    top_k: int = 5
):
    results = search_chunks(
        query=query,
        db=db,
        top_k=top_k
    )

    context = []

    for chunk, distance in results:
        context.append({
            "content": chunk.content,
            "document_id": chunk.document_id,
            "distance": float(distance)
        })

    return context 