import requests
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class EmbeddingResult:
    embedding: List[float]
    dimension: int


class PostfinderClient:
    """Client for the Postfinder embedding API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.

        Args:
            base_url: Server URL (e.g., "http://localhost:8000").
        """
        self.base_url = base_url.rstrip("/")

    def embed_query(self, query: str, dimension: Optional[int] = None) -> EmbeddingResult:
        """Embed a search query.

        Args:
            query: The search query text.
            dimension: Optional MRL dimension (64, 128, 256, 512, 1024, 2048).
        """
        payload = {"query": query}
        if dimension:
            payload["dimension"] = dimension
        response = requests.post(
            f"{self.base_url}/embed_query",
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        return EmbeddingResult(embedding=data["embedding"], dimension=data["dimension"])

    def embed_post(
        self,
        caption: Optional[str] = None,
        image_url: Optional[str] = None,
        image_urls: Optional[List[str]] = None,
        dimension: Optional[int] = None,
    ) -> EmbeddingResult:
        """Embed an Instagram post (caption and/or images).

        Args:
            caption: The post caption text.
            image_url: URL of a single image (convenience param).
            image_urls: List of image URLs.
            dimension: Optional MRL dimension (64, 128, 256, 512, 1024, 2048).
        """
        urls = []
        if image_url:
            urls.append(image_url)
        if image_urls:
            urls.extend(image_urls)

        payload = {
            "post": {
                "caption": caption,
                "image_urls": urls,
            }
        }
        if dimension:
            payload["dimension"] = dimension

        response = requests.post(
            f"{self.base_url}/embed_post",
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        return EmbeddingResult(embedding=data["embedding"], dimension=data["dimension"])

    def embed_posts(
        self,
        posts: List[dict],
        dimension: Optional[int] = None,
    ) -> List[EmbeddingResult]:
        """Embed multiple posts in batch.

        Args:
            posts: List of post dicts with 'caption' and 'image_urls' keys.
            dimension: Optional MRL dimension (64, 128, 256, 512, 1024, 2048).
        """
        payload = {"posts": posts}
        if dimension:
            payload["dimension"] = dimension

        response = requests.post(
            f"{self.base_url}/embed_posts",
            json=payload,
            timeout=600,
        )
        response.raise_for_status()
        data = response.json()
        return [
            EmbeddingResult(embedding=emb, dimension=data["dimension"])
            for emb in data["embeddings"]
        ]

    def health(self) -> dict:
        """Check API health."""
        response = requests.get(
            f"{self.base_url}/health",
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        import numpy as np

        a = np.array(embedding1)
        b = np.array(embedding2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
