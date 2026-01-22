"""
FastAPI server for Qwen3-VL-Embedding-2B on Mac Studio (Apple Silicon).
Run with: uv run uvicorn server_local:app --host 0.0.0.0 --port 8000
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.models import (
    PostEmbedRequest,
    BatchPostEmbedRequest,
    QueryEmbedRequest,
    EmbeddingResponse,
    BatchEmbeddingResponse,
    HealthResponse,
)

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLPreTrainedModel,
    Qwen3VLModel,
    Qwen3VLConfig,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.modeling_outputs import ModelOutput
from qwen_vl_utils import process_vision_info


MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"


# === Model classes (from example.py) ===
@dataclass
class Qwen3VLForEmbeddingOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None


class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    config: Qwen3VLConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        return Qwen3VLForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


# === Embedding service ===
class EmbeddingService:
    def __init__(self):
        # Detect device: MPS for Mac, CUDA for NVIDIA, else CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Apple Silicon)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        self.default_instruction = "Represent the user's input."

        print(f"Loading model {MODEL_ID}...")
        self.model = Qwen3VLForEmbedding.from_pretrained(
            MODEL_ID, trust_remote_code=True, dtype=torch.float16
        ).to(self.device)
        self.processor = Qwen3VLProcessor.from_pretrained(MODEL_ID, padding_side="right")
        self.model.eval()

        # Compile model for faster inference (if supported)
        if hasattr(torch, "compile") and self.device.type != "mps":
            # torch.compile not fully supported on MPS yet
            self.model = torch.compile(self.model, mode="reduce-overhead")
            print("Model compiled with torch.compile")

        print("Model loaded.")

    @staticmethod
    def _pooling_last(hidden_state, attention_mask):
        flipped_tensor = attention_mask.flip(dims=[1])
        last_one_positions = flipped_tensor.argmax(dim=1)
        col = attention_mask.shape[1] - last_one_positions - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    def _forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return {
            "last_hidden_state": outputs.last_hidden_state,
            "attention_mask": inputs.get("attention_mask"),
        }

    def _format_input(
        self,
        text: Optional[str] = None,
        image: Optional[str] = None,
        images: Optional[List[str]] = None,
    ) -> List[Dict]:
        content = []
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": self.default_instruction}]},
            {"role": "user", "content": content},
        ]

        # Combine single image and image list
        all_images = []
        if image:
            all_images.append(image)
        if images:
            all_images.extend(images)

        if not text and not all_images:
            content.append({"type": "text", "text": "NULL"})
            return conversation

        for img_url in all_images:
            content.append({
                "type": "image",
                "image": img_url,
                "min_pixels": 4096,
                "max_pixels": 1843200,
            })

        if text:
            content.append({"type": "text", "text": text})

        return conversation

    def get_embedding(
        self,
        text: Optional[str] = None,
        image: Optional[str] = None,
        images: Optional[List[str]] = None,
        dimension: Optional[int] = None,
    ) -> List[float]:
        conversation = self._format_input(text=text, image=image, images=images)

        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        try:
            processed_images, video_inputs, video_kwargs = process_vision_info(
                conversation,
                image_patch_size=16,
                return_video_metadata=True,
                return_video_kwargs=True,
            )
        except Exception:
            processed_images = None
            video_inputs = None
            video_kwargs = {"do_sample_frames": False}

        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos = list(videos)
            video_metadata = list(video_metadata)
        else:
            videos, video_metadata = None, None

        inputs = self.processor(
            text=prompt,
            images=processed_images,
            videos=videos,
            video_metadata=video_metadata,
            truncation=True,
            max_length=8192,
            padding=True,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._forward(inputs)
        embedding = self._pooling_last(outputs["last_hidden_state"], outputs["attention_mask"])
        embedding = F.normalize(embedding, p=2, dim=-1)
        embedding = embedding.squeeze().cpu().float().numpy()

        # MRL: reduce dimension if requested
        if dimension is not None and dimension < len(embedding):
            embedding = embedding[:dimension]
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding.tolist()


# === FastAPI app ===
embedding_service: Optional[EmbeddingService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_service
    embedding_service = EmbeddingService()
    yield
    # Cleanup if needed


app = FastAPI(title="Postfinder Embedding API", lifespan=lifespan)


@app.get("/health")
def health() -> HealthResponse:
    return HealthResponse(status="ok", model=MODEL_ID)


@app.post("/embed_query")
def embed_query(request: QueryEmbedRequest) -> EmbeddingResponse:
    """Generate embedding for a search query."""
    embedding = embedding_service.get_embedding(
        text=request.query,
        dimension=request.dimension,
    )
    return EmbeddingResponse(embedding=embedding, dimension=len(embedding))


@app.post("/embed_post")
def embed_post(request: PostEmbedRequest) -> EmbeddingResponse:
    """Generate embedding for a single post (caption + multiple images)."""
    embedding = embedding_service.get_embedding(
        text=request.post.caption,
        images=request.post.image_urls,
        dimension=request.dimension,
    )
    return EmbeddingResponse(embedding=embedding, dimension=len(embedding))


@app.post("/embed_posts")
def embed_posts(request: BatchPostEmbedRequest) -> BatchEmbeddingResponse:
    """Generate embeddings for multiple posts."""
    embeddings = []
    for post in request.posts:
        embedding = embedding_service.get_embedding(
            text=post.caption,
            images=post.image_urls,
            dimension=request.dimension,
        )
        embeddings.append(embedding)

    dimension = len(embeddings[0]) if embeddings else 0
    return BatchEmbeddingResponse(
        embeddings=embeddings,
        dimension=dimension,
        count=len(embeddings),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
