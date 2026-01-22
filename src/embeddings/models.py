"""SQLModel definitions for embedding generation."""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, PrimaryKeyConstraint, String, Text, func
from sqlmodel import Column, Field, SQLModel


class CorePost(SQLModel, table=True):
    """Instagram post model."""

    __tablename__ = "core_posts"
    __table_args__ = {"schema": "instagram"}

    ig_media_id: str = Field(sa_column=Column(String(50), primary_key=True))
    shortcode: str | None = Field(default=None, sa_column=Column(String(50)))
    ig_user_id: str = Field(sa_column=Column(String(50)))
    caption: str | None = Field(default=None, sa_column=Column(Text))
    timestamp: datetime | None = Field(default=None, sa_column=Column(DateTime))
    media_type: str | None = Field(default=None, sa_column=Column(String(20)))
    media_product_type: str | None = Field(default=None, sa_column=Column(String(20)))
    followed_post: bool = Field(default=False, sa_column=Column(Boolean, default=False))
    suspected: bool = Field(default=False, sa_column=Column(Boolean, default=False))
    authors_checked: bool = Field(default=False, sa_column=Column(Boolean, default=False))
    inserted_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime, server_default=func.date_trunc("second", func.now()))
    )


class CorePostMedia(SQLModel, table=True):
    """Instagram post media model."""

    __tablename__ = "core_post_media"
    __table_args__ = (
        PrimaryKeyConstraint("ig_media_id", "media_order"),
        {"schema": "instagram"},
    )

    ig_media_id: str = Field(sa_column=Column(String(50)))
    media_order: int = Field(sa_column=Column(Integer))
    parent_ig_media_id: str | None = Field(default=None, sa_column=Column(String(50)))
    media_type: str | None = Field(default=None, sa_column=Column(String(20)))
    media_url: str | None = Field(default=None, sa_column=Column(Text))
    video_url: str | None = Field(default=None, sa_column=Column(Text))
    thumbnail_url: str | None = Field(default=None, sa_column=Column(Text))


class PostEmbedding(SQLModel, table=True):
    """Post embedding model - stores vector embeddings for posts."""

    __tablename__ = "post_embeddings"
    __table_args__ = {"schema": "instagram"}

    ig_media_id: str = Field(
        sa_column=Column(
            String(50),
            ForeignKey("instagram.core_posts.ig_media_id"),
            primary_key=True,
        )
    )
    embedding: list[float] = Field(sa_column=Column(Vector(512), nullable=False))
    created_at: datetime | None = Field(
        default=None, sa_column=Column(DateTime, server_default=func.now())
    )
