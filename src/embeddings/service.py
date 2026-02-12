"""Embedding service with async batch processing."""

import asyncio
import hashlib
import time
from typing import List, Optional

from loguru import logger
from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.chunking.chunker import TextChunker
from src.config import get_settings
from src.embeddings.cache import EmbeddingCache
from src.models import Chunk
from src.observability.metrics import (
    embedding_requests_total,
    embedding_tokens_total,
    embedding_duration,
    embedding_cost,
)


class EmbeddingService:
    """Async embedding generation with caching and cost tracking."""

    # OpenAI pricing (as of 2025)
    PRICING = {
        "text-embedding-3-large": 0.00013 / 1000,  # per token
        "text-embedding-3-small": 0.00002 / 1000,
        "text-embedding-ada-002": 0.00010 / 1000,
    }

    def __init__(self):
        """Initialize embedding service."""
        self.settings = get_settings()
        self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        self.cache = EmbeddingCache()
        self.model = self.settings.openai_embedding_model
        self.dimensions = self.settings.openai_embedding_dimensions
        
        logger.info(f"EmbeddingService initialized with model: {self.model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
    )
    async def embed_text(self, text: str, use_cache: bool = True) -> List[float]:
        """Generate embedding for single text.

        Args:
            text: Input text
            use_cache: Whether to use cache

        Returns:
            Embedding vector
        """
        # Compute hash
        content_hash = self._compute_hash(text)

        # Check cache
        if use_cache:
            cached = self.cache.get(content_hash, self.model)
            if cached is not None:
                embedding_requests_total.labels(
                    model=self.model, cached="true"
                ).inc()
                return cached

        # Generate embedding
        start_time = time.time()
        
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions
            )
            
            embedding = response.data[0].embedding
            tokens_used = response.usage.total_tokens
            
            duration = time.time() - start_time
            
            # Calculate cost
            cost = self._calculate_cost(tokens_used)
            
            # Update metrics
            embedding_requests_total.labels(
                model=self.model, cached="false"
            ).inc()
            embedding_tokens_total.labels(model=self.model).inc(tokens_used)
            embedding_duration.labels(
                model=self.model, batch_size="1"
            ).observe(duration)
            embedding_cost.labels(model=self.model).observe(cost)
            
            # Cache result
            if use_cache:
                self.cache.set(
                    content_hash=content_hash,
                    embedding=embedding,
                    model=self.model,
                    dimensions=self.dimensions,
                    token_count=tokens_used
                )
            
            logger.debug(
                f"Generated embedding: {len(embedding)} dims, "
                f"{tokens_used} tokens, {duration:.2f}s, ${cost:.6f}"
            )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    async def embed_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
        batch_size: int = 100
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts with batching.

        Args:
            texts: List of texts
            use_cache: Whether to use cache
            batch_size: Maximum batch size for API calls

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Compute hashes
        hashes = [self._compute_hash(text) for text in texts]
        
        # Check cache for batch
        cached_embeddings = {}
        if use_cache:
            cached_embeddings = self.cache.get_batch(hashes, self.model)
        
        # Identify texts that need embedding
        texts_to_embed = []
        text_indices = []
        
        for i, (text, content_hash) in enumerate(zip(texts, hashes)):
            if content_hash not in cached_embeddings:
                texts_to_embed.append(text)
                text_indices.append(i)
        
        logger.info(
            f"Cache stats: {len(cached_embeddings)} hits, "
            f"{len(texts_to_embed)} misses"
        )
        
        # Generate embeddings for uncached texts
        new_embeddings = []
        if texts_to_embed:
            new_embeddings = await self._embed_batch_uncached(
                texts_to_embed, batch_size
            )
            
            # Cache new embeddings
            if use_cache:
                for text, embedding in zip(texts_to_embed, new_embeddings):
                    content_hash = self._compute_hash(text)
                    # Estimate token count
                    token_count = len(text) // 4
                    self.cache.set(
                        content_hash=content_hash,
                        embedding=embedding,
                        model=self.model,
                        dimensions=self.dimensions,
                        token_count=token_count
                    )
        
        # Reconstruct full results in original order
        results = [None] * len(texts)
        
        # Fill in cached results
        for i, content_hash in enumerate(hashes):
            if content_hash in cached_embeddings:
                results[i] = cached_embeddings[content_hash]
        
        # Fill in new results
        for i, embedding in zip(text_indices, new_embeddings):
            results[i] = embedding
        
        return results

    async def _embed_batch_uncached(
        self,
        texts: List[str],
        batch_size: int
    ) -> List[List[float]]:
        """Generate embeddings without cache (internal helper).

        Args:
            texts: Texts to embed
            batch_size: Batch size

        Returns:
            List of embeddings
        """
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            start_time = time.time()
            
            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimensions
                )
                
                # Extract embeddings in order
                embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings)
                
                tokens_used = response.usage.total_tokens
                duration = time.time() - start_time
                cost = self._calculate_cost(tokens_used)
                
                # Update metrics
                embedding_requests_total.labels(
                    model=self.model, cached="false"
                ).inc(len(batch))
                embedding_tokens_total.labels(model=self.model).inc(tokens_used)
                embedding_duration.labels(
                    model=self.model, batch_size=str(len(batch))
                ).observe(duration)
                embedding_cost.labels(model=self.model).observe(cost)
                
                logger.info(
                    f"Batch {i//batch_size + 1}: {len(batch)} embeddings, "
                    f"{tokens_used} tokens, {duration:.2f}s, ${cost:.6f}"
                )
                
                # Small delay to avoid rate limits
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in batch {i//batch_size + 1}: {e}")
                raise
        
        return all_embeddings

    async def embed_chunks(
        self,
        chunks: List[Chunk],
        use_cache: bool = True,
        batch_size: int = 100
    ) -> List[Chunk]:
        """Generate embeddings for chunks and update them.

        Args:
            chunks: List of chunks
            use_cache: Whether to use cache
            batch_size: Batch size

        Returns:
            Chunks with embeddings populated
        """
        if not chunks:
            return []

        logger.info(f"Embedding {len(chunks)} chunks")
        
        # Extract texts
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        embeddings = await self.embed_batch(texts, use_cache, batch_size)
        
        # Update chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            chunk.embedding_model = self.model
        
        logger.info(f"Successfully embedded {len(chunks)} chunks")
        
        return chunks

    def _calculate_cost(self, tokens: int) -> float:
        """Calculate cost for tokens.

        Args:
            tokens: Token count

        Returns:
            Cost in USD
        """
        price_per_token = self.PRICING.get(self.model, 0.00013 / 1000)
        return tokens * price_per_token

    @staticmethod
    def _compute_hash(text: str) -> str:
        """Compute SHA-256 hash of text.

        Args:
            text: Input text

        Returns:
            Hash string
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()