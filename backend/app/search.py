from sqlalchemy.orm import Session

from .models import DocumentChunk
from .embeddings import generate_embedding

def search_chunks(
    query: str,
    db: Session,
    top_k: int = 5
):
    query_embedding = generate_embedding(query)

    results = (
        db.query(
            DocumentChunk,
            DocumentChunk.embedding.cosine_distance(query_embedding).label(
                "distance"
            )
        )
        .filter(DocumentChunk.embedding.is_not(None))
        .order_by("distance")
        .limit(top_k)
        .all()
    )

    return results