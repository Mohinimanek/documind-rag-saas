from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
import os
from pypdf import PdfReader

from .database import get_db
from .models import Document, DocumentChunk
from .chunking import chunk_text
from .embeddings import generate_embedding
from .search import search_chunks
from .rag import retrieve_context
from .llm import generate_answer

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class User(BaseModel):
    name:str

class DocumentCreate(BaseModel):
    title:str
    description:str

class AskRequest(BaseModel):
    question: str
    top_k: int = 5

@app.get("/")
def home():
    return{"message":"DocuMind is ready!"}

@app.get("/health")
def health_check():
    return{"status":"healthy"}

@app.get("/about")
def about():
    return{"project":"DocuMind","Version":"1.0"}

@app.post("/hello")
def hello_user(user:User):
    return{"message":f"Hello, {user.name}"}

@app.post("/documents")
def create_document(document:DocumentCreate, db: Session = Depends(get_db)):
    new_document = Document(
        title=document.title,
        description=document.description
    )

    db.add(new_document)
    db.commit()
    db.refresh(new_document)

    return{
        "message":"Document received",
        "title":document.title,
        "description":document.description
    }

@app.get("/documents")
def get_documents(db:Session = Depends(get_db)):
    documents = db.query(Document).all()
    return documents 

@app.put("/documents/{document_id}")
def update_document(
    document_id: int,
    document: DocumentCreate,
    db: Session = Depends(get_db)
):
    existing_document = db.query(Document).filter(
        Document.id == document_id
    ).first()

    if not existing_document:
        return{"message":"Document not found"}

    existing_document.title = document.title
    existing_document.description = document.description

    db.commit()
    db.refresh(existing_document)

    return{
        "message":"Document updated",
        "id":existing_document.id,
        "title":existing_document.title,
        "description":existing_document.description
    }

@app.delete("/documents/{document_id}")
def delete_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    existing_document = db.query(Document).filter(
        Document.id == document_id
    ).first()

    if not existing_document:
        return{"message":"Document not found"}

    db.delete(existing_document)
    db.commit()

    return{
        "message":"Document deleted",
        "id": document_id
    }

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    new_document = Document(
        title=file.filename,
        description="Uploaded PDF document",
        content=text
    )

    db.add(new_document)
    db.commit()
    db.refresh(new_document)

    chunks = chunk_text(text)

    for chunk in chunks:
        embedding  = generate_embedding(chunk)
        
        new_chunk = DocumentChunk(
            document_id=new_document.id,
            content=chunk,
            embedding=embedding
        )

        db.add(new_chunk)

    db.commit()

    return{
        "message":"Document uploaded successfully",
        "document_id":new_document.id,
        "filename":file.filename,
        "pages":len(reader.pages),
        "characters":len(text),
        "chunks_created": len(chunks),
        "text_preview":text[:500]
    }

@app.get("/search")
def search_documents(
    query: str,
    top_k: int = 5,
    db: Session = Depends(get_db)
):
    resutls = search_chunks(
        query=query,
        db=db,
        top_k=top_k
    )

    return[
        {
            "chunk_id": result_chunk.id,
            "document_id": result_chunk.document_id,
            "content": result_chunk.content,
            "distance": float(distance)
        }
        for result_chunk, distance in resutls
    ]

@app.post("/ask")
def ask_question(
    request: AskRequest,
    db: Session = Depends(get_db)
):
    context = retrieve_context(
        query=request.question,
        db=db,
        top_k=request.top_k
    )

    answer = generate_answer(
        question=request.question,
        context=context
    )

    return{
        "question":request.question,
        "answer":answer,
        "sources":context
    }