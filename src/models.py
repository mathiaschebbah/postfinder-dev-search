"""Pydantic models for the Postfinder Embedding API."""

from typing import Optional, List
from pydantic import BaseModel, Field


# === Core Models ===


class Post(BaseModel):
    """A social media post with caption and images."""

    caption: Optional[str] = Field(default=None, description="The post caption text")
    image_urls: List[str] = Field(
        default_factory=list, description="List of image URLs for the post"
    )


# === Request Models ===


class PostEmbedRequest(BaseModel):
    """Request to embed a single post."""

    post: Post
    dimension: Optional[int] = Field(
        default=None, description="MRL dimension: 64, 128, 256, 512, 1024, 2048"
    )


class BatchPostEmbedRequest(BaseModel):
    """Request to embed multiple posts."""

    posts: List[Post]
    dimension: Optional[int] = Field(
        default=None, description="MRL dimension: 64, 128, 256, 512, 1024, 2048"
    )


class QueryEmbedRequest(BaseModel):
    """Request to embed a search query."""

    query: str
    dimension: Optional[int] = Field(
        default=None, description="MRL dimension: 64, 128, 256, 512, 1024, 2048"
    )


# === Response Models ===


class EmbeddingResponse(BaseModel):
    """Response containing a single embedding."""

    embedding: List[float]
    dimension: int


class BatchEmbeddingResponse(BaseModel):
    """Response containing multiple embeddings."""

    embeddings: List[List[float]]
    dimension: int
    count: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model: str
