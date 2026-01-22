"""Generate embeddings for Instagram posts using Cloud Run Job parallelism.

Cloud Run Job environment variables (auto-set):
- CLOUD_RUN_TASK_INDEX: Index of this task (0, 1, 2, ...)
- CLOUD_RUN_TASK_COUNT: Total number of parallel tasks

Required environment variables:
- DATABASE_URL: PostgreSQL connection string

Optional environment variables:
- EMBEDDING_API_URL: Embedding API URL (default: https://embedding.views.fr)
- EMBEDDING_DIMENSION: Vector dimension (default: 1024)
- BATCH_SIZE: Posts per API call (default: 10)
"""

import asyncio
import os
import sys
import time

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import select

from src.client import PostfinderClient
from src.embeddings.models import CorePost, CorePostMedia, PostEmbedding

# Cloud Run Job environment
TASK_INDEX = int(os.environ.get("CLOUD_RUN_TASK_INDEX", 0))
TASK_COUNT = int(os.environ.get("CLOUD_RUN_TASK_COUNT", 1))

# Configuration from environment
DATABASE_URL = os.environ.get("DATABASE_URL")
EMBEDDING_API_URL = os.environ.get("EMBEDDING_API_URL", "https://embedding.views.fr")
EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", 512))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 10))


def log(message: str, level: str = "INFO"):
    """Print structured log for Cloud Run."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    task_info = f"[task {TASK_INDEX}/{TASK_COUNT}]"
    print(f"[{timestamp}] [{level}] {task_info} {message}")


def log_separator():
    """Print a visual separator."""
    print("=" * 80)


def create_engine():
    """Create async database engine."""
    if not DATABASE_URL:
        log("DATABASE_URL environment variable is required", level="ERROR")
        sys.exit(1)

    return create_async_engine(
        DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
        connect_args={"statement_cache_size": 0},
    )


async def count_posts_without_embeddings(session: AsyncSession) -> int:
    """Count total posts that need embeddings."""
    embedded_posts = select(PostEmbedding.ig_media_id).subquery()

    stmt = (
        select(func.count())
        .select_from(CorePost)
        .outerjoin(embedded_posts, CorePost.ig_media_id == embedded_posts.c.ig_media_id)
        .where(embedded_posts.c.ig_media_id.is_(None))
        .where(CorePost.suspected.is_(False))
        .where(CorePost.media_product_type != "STORY")
    )

    result = await session.execute(stmt)
    return result.scalar() or 0


async def fetch_posts_for_task(
    session: AsyncSession,
    task_index: int,
    task_count: int,
) -> list[dict]:
    """Fetch posts assigned to this task using modulo partitioning.

    Each task processes posts where: row_number % task_count == task_index
    This ensures even distribution without needing to know total count upfront.
    """
    log(f"Fetching posts for task {task_index} (of {task_count} total tasks)...")

    # Subquery: posts that already have embeddings
    embedded_posts = select(PostEmbedding.ig_media_id).subquery()

    # Subquery: first media (media_order = 0) for carousel/video thumbnail
    first_media = (
        select(
            CorePostMedia.parent_ig_media_id,
            CorePostMedia.media_type,
            CorePostMedia.media_url,
            CorePostMedia.thumbnail_url,
        )
        .where(CorePostMedia.media_order == 0)
        .subquery()
    )

    # Get all posts without embeddings (ordered for deterministic partitioning)
    from sqlalchemy import case

    image_url_expr = case(
        (first_media.c.media_type == "VIDEO", first_media.c.thumbnail_url),
        else_=first_media.c.media_url,
    )

    stmt = (
        select(
            CorePost.ig_media_id,
            CorePost.media_type,
            CorePost.caption,
            image_url_expr.label("image_url"),
        )
        .outerjoin(embedded_posts, CorePost.ig_media_id == embedded_posts.c.ig_media_id)
        .outerjoin(first_media, first_media.c.parent_ig_media_id == CorePost.ig_media_id)
        .where(embedded_posts.c.ig_media_id.is_(None))
        .where(CorePost.suspected.is_(False))
        .where(CorePost.media_product_type != "STORY")
        .order_by(CorePost.timestamp.desc().nulls_last())  # Most recent first
    )

    result = await session.execute(stmt)
    all_rows = result.fetchall()

    # Partition by modulo: this task handles rows where index % task_count == task_index
    posts = []
    for i, row in enumerate(all_rows):
        if i % task_count == task_index:
            posts.append({
                "ig_media_id": row.ig_media_id,
                "media_type": row.media_type,
                "caption": row.caption,
                "image_url": row.image_url,
            })

    log(f"Task {task_index} will process {len(posts)} posts (from {len(all_rows)} total)")
    return posts


def process_batch(
    client: PostfinderClient,
    posts: list[dict],
    batch_num: int,
    total_batches: int,
) -> list[tuple[str, list[float]]]:
    """Process a batch of posts using batch API. Returns list of (ig_media_id, embedding)."""
    log(f"BATCH {batch_num}/{total_batches} - Processing {len(posts)} posts")

    # Prepare batch payload for API
    api_posts = [
        {
            "caption": post["caption"],
            "image_urls": [post["image_url"]] if post["image_url"] else [],
        }
        for post in posts
    ]

    start_time = time.time()
    results = client.embed_posts(posts=api_posts, dimension=EMBEDDING_DIMENSION)
    api_time = time.time() - start_time

    log(f"API returned {len(results)} embeddings in {api_time:.2f}s ({api_time/len(posts):.2f}s/post)")

    return [
        (post["ig_media_id"], result.embedding)
        for post, result in zip(posts, results)
    ]


def save_embedding(session: AsyncSession, ig_media_id: str, embedding: list[float]):
    """Add a new embedding to the session (no commit)."""
    post_embedding = PostEmbedding(
        ig_media_id=ig_media_id,
        embedding=embedding,
    )
    session.add(post_embedding)


async def main():
    """Main function for Cloud Run Job."""
    log_separator()
    log("POSTFINDER EMBEDDING GENERATOR - Cloud Run Job")
    log_separator()
    log(f"Configuration:")
    log(f"  Task: {TASK_INDEX + 1} of {TASK_COUNT}")
    log(f"  Embedding API: {EMBEDDING_API_URL}")
    log(f"  Embedding dimension: {EMBEDDING_DIMENSION}")
    log(f"  Batch size: {BATCH_SIZE}")
    log_separator()

    # Initialize client
    log("Initializing embedding API client...")
    client = PostfinderClient(base_url=EMBEDDING_API_URL)

    # Check API health
    try:
        health = client.health()
        log(f"API health: status={health.get('status')}, model={health.get('model')}")
    except Exception as e:
        log(f"WARNING: Could not check API health: {e}", level="WARN")

    # Create database engine
    engine = create_engine()
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    total_start_time = time.time()

    async with async_session() as session:
        # Count total posts (for logging)
        total_count = await count_posts_without_embeddings(session)
        log(f"Total posts without embeddings: {total_count}")

        # Fetch posts for this task
        posts = await fetch_posts_for_task(session, TASK_INDEX, TASK_COUNT)

        if not posts:
            log("No posts assigned to this task. Exiting.")
            return

        # Calculate batches
        num_batches = (len(posts) + BATCH_SIZE - 1) // BATCH_SIZE
        log(f"Will process {len(posts)} posts in {num_batches} batch(es)")
        log_separator()

        # Process batches (commit after each batch for incremental progress)
        total_success = 0
        total_errors = 0

        for batch_num in range(1, num_batches + 1):
            start_idx = (batch_num - 1) * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(posts))
            batch_posts = posts[start_idx:end_idx]

            try:
                embeddings = process_batch(client, batch_posts, batch_num, num_batches)

                # Save and commit this batch immediately
                for ig_media_id, embedding in embeddings:
                    save_embedding(session, ig_media_id, embedding)
                await session.commit()

                total_success += len(embeddings)
                log(f"BATCH {batch_num}/{num_batches} - Committed {len(embeddings)} embeddings (total: {total_success}/{len(posts)})")

            except Exception as e:
                log(f"Batch {batch_num} FAILED: {e}", level="ERROR")
                await session.rollback()
                total_errors += len(batch_posts)

    # Cleanup
    await engine.dispose()

    # Summary
    total_time = time.time() - total_start_time
    log_separator()
    log("SUMMARY")
    log_separator()
    log(f"Task {TASK_INDEX + 1}/{TASK_COUNT} completed")
    log(f"Posts processed: {total_success + total_errors}")
    log(f"  Success: {total_success}")
    log(f"  Errors: {total_errors}")
    log(f"Total time: {total_time:.2f}s")
    if total_success > 0:
        log(f"Average time per post: {total_time / total_success:.2f}s")
    log_separator()
    log("Done!")


if __name__ == "__main__":
    asyncio.run(main())
