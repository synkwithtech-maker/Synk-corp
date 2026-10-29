"""
RAG document processing and retrieval schemas.
"""

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class IngestUrlRequest(BaseModel):
    url: str = Field(
        ...,
        min_length=8,
        description="Target webpage URL to crawl and index",
        examples=["https://en.wikipedia.org/wiki/Artificial_intelligence"],
    )


class IngestResponse(BaseModel):
    status: str
    source: str
    chunks_indexed: int
    message: str


class DocumentMetadata(BaseModel):
    filename: str
    file_type: str
    size_kb: float
    last_modified: str


class DocumentListResponse(BaseModel):
    total_files: int
    documents: List[DocumentMetadata]


class RAGUploadResponse(BaseModel):
    status: str
    filename: str
    file_type: str
    file_size_kb: float
    chunks_indexed: int
    preview_chunks: List[str]
    ready_for_chat: bool
    message: str


class UpdateDocumentResponse(BaseModel):
    status: str
    filename: str
    file_type: str
    file_size_kb: float
    old_vectors_removed: bool
    new_chunks_indexed: int
    preview_chunks: List[str]
    ready_for_chat: bool
    message: str


class DeleteDocumentResponse(BaseModel):
    status: str
    filename: str
    file_deleted: bool
    vectors_deleted: bool
    message: str


class RAGBatchUploadResponse(BaseModel):
    status: str
    files_processed: int
    total_chunks_indexed: int
    results: List[Dict[str, Any]]
    message: str


class RAGQueryTestRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Test search query against knowledge base", examples=["refund policy"])
    top_k: int = Field(default=3, ge=1, le=10, description="Number of context chunks to retrieve")


class RetrievedChunk(BaseModel):
    source: str
    page_content: str
    metadata: Dict[str, Any]


class RAGQueryTestResponse(BaseModel):
    query: str
    total_chunks_found: int
    chunks: List[RetrievedChunk]
