import requests
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class EmbeddingResult:
    embedding: List[float]
    dimension: int


class PostfinderClient:
    """Client for the Postfinder embedding API."""

    # Dev mode URLs (from modal serve)
    DEV_URLS = {
        "embed-query": "https://mathias-chebbah--postfinder-embed-v3-embeddingmodel--f5c7cc-dev.modal.run",
        "embed-post": "https://mathias-chebbah--postfinder-embed-v3-embeddingmodel--a69350-dev.modal.run",
        "health": "https://mathias-chebbah--postfinder-embed-v3-embeddingmodel-health-dev.modal.run",
    }

    # Production URLs (from modal deploy)
    PROD_URLS = {
        "embed-query": "https://mathias-chebbah--postfinder-embed-v3-embeddingmodel-embed-query.modal.run",
        "embed-post": "https://mathias-chebbah--postfinder-embed-v3-embeddingmodel-embed-post.modal.run",
        "health": "https://mathias-chebbah--postfinder-embed-v3-embeddingmodel-health.modal.run",
    }

    def __init__(self, base_url: Optional[str] = None, dev_mode: bool = True):
        """
        Initialize the client.

        Args:
            base_url: Local server URL (e.g., "http://localhost:8000" or Cloudflare tunnel URL).
                      If provided, ignores dev_mode and uses this URL directly.
            dev_mode: If True and base_url is None, use Modal dev URLs. Otherwise use Modal prod URLs.
        """
        self.base_url = base_url
        self.dev_mode = dev_mode

    def _url(self, endpoint: str) -> str:
        if self.base_url:
            # Local server: endpoints are /health, /embed_query, /embed_post, /embed_posts
            endpoint_map = {
                "embed-query": "/embed_query",
                "embed-post": "/embed_post",
                "embed-posts": "/embed_posts",
                "health": "/health",
            }
            return f"{self.base_url.rstrip('/')}{endpoint_map[endpoint]}"
        else:
            # Modal URLs
            urls = self.DEV_URLS if self.dev_mode else self.PROD_URLS
            return urls[endpoint]

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
            self._url("embed-query"),
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
        # Build image_urls list
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
            self._url("embed-post"),
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
            self._url("embed-posts"),
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
            self._url("health"),
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
