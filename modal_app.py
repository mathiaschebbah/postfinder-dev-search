import modal
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

app = modal.App("postfinder-embed-v3")

modal_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.57.0",
        "qwen-vl-utils>=0.0.14",
        "numpy>=1.26.0",
        "pillow>=10.0.0",
        "fastapi>=0.115.0",
        "accelerate>=0.26.0",
        "pydantic>=2.0.0",
    )
)

MODEL_ID = "Qwen/Qwen3-VL-Embedding-8B"


# === Pydantic models (duplicated for Modal compatibility) ===


class Post(BaseModel):
    """A social media post with caption and images."""

    caption: Optional[str] = Field(default=None, description="The post caption text")
    image_urls: List[str] = Field(
        default_factory=list, description="List of image URLs for the post"
    )


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


@app.cls(
    image=modal_image,
    gpu="L4",
    timeout=600,
    scaledown_window=300,
)
class EmbeddingModel:
    @modal.enter()
    def load_model(self):
        import torch
        from dataclasses import dataclass
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLPreTrainedModel,
            Qwen3VLModel,
            Qwen3VLConfig,
        )
        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
        from transformers.modeling_outputs import ModelOutput

        # === Copied from example.py ===
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
        # === End copy ===

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.default_instruction = "Represent the user's input."

        print(f"Loading model {MODEL_ID}...")
        self.model = Qwen3VLForEmbedding.from_pretrained(
            MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16
        ).to(self.device)
        self.processor = Qwen3VLProcessor.from_pretrained(MODEL_ID, padding_side="right")
        self.model.eval()
        print("Model loaded.")

    # === Copied from example.py ===
    @staticmethod
    def _pooling_last(hidden_state, attention_mask):
        import torch
        flipped_tensor = attention_mask.flip(dims=[1])
        last_one_positions = flipped_tensor.argmax(dim=1)
        col = attention_mask.shape[1] - last_one_positions - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    def _forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import torch
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
        """Format input like example.py format_model_input()"""
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
            # Match example.py: include min_pixels and max_pixels
            # IMAGE_FACTOR = 16 * 2 = 32
            # MIN_PIXELS = 4 * 32 * 32 = 4096
            # MAX_PIXELS = 1800 * 32 * 32 = 1843200
            content.append({
                "type": "image",
                "image": img_url,
                "min_pixels": 4096,
                "max_pixels": 1843200,
            })

        if text:
            content.append({"type": "text", "text": text})

        return conversation

    def _get_embedding(
        self,
        text: Optional[str] = None,
        image: Optional[str] = None,
        images: Optional[List[str]] = None,
        dimension: Optional[int] = None,
    ) -> List[float]:
        import torch
        import torch.nn.functional as F
        import numpy as np
        from qwen_vl_utils import process_vision_info

        conversation = self._format_input(text=text, image=image, images=images)

        # Apply chat template (like example.py _preprocess_inputs)
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        # Process vision info with same params as example.py
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

        # Tokenize (like example.py)
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

        # Forward pass
        outputs = self._forward(inputs)

        # Pooling
        embedding = self._pooling_last(outputs["last_hidden_state"], outputs["attention_mask"])

        # Normalize
        embedding = F.normalize(embedding, p=2, dim=-1)
        embedding = embedding.squeeze().cpu().float().numpy()

        # MRL: reduce dimension if requested (like example.py reduce_embedding_dim)
        if dimension is not None and dimension < len(embedding):
            embedding = embedding[:dimension]
            # Re-normalize after truncation
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding.tolist()
    # === End copy ===

    @modal.fastapi_endpoint(method="POST")
    def embed_post(self, request: PostEmbedRequest) -> EmbeddingResponse:
        """Embed a single post (caption + multiple images)."""
        embedding = self._get_embedding(
            text=request.post.caption,
            images=request.post.image_urls,
            dimension=request.dimension,
        )
        return EmbeddingResponse(embedding=embedding, dimension=len(embedding))

    @modal.fastapi_endpoint(method="POST")
    def embed_posts(self, request: BatchPostEmbedRequest) -> BatchEmbeddingResponse:
        """Embed multiple posts."""
        embeddings = []
        for post in request.posts:
            embedding = self._get_embedding(
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

    @modal.fastapi_endpoint(method="POST")
    def embed_query(self, request: QueryEmbedRequest) -> EmbeddingResponse:
        """Embed a search query."""
        embedding = self._get_embedding(
            text=request.query,
            dimension=request.dimension,
        )
        return EmbeddingResponse(embedding=embedding, dimension=len(embedding))

    @modal.fastapi_endpoint(method="GET")
    def health(self) -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(status="ok", model=MODEL_ID)
