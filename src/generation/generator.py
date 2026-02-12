"""Response generation with citations and confidence scoring."""

import json
import time
from typing import List, Optional, Dict, Any
from uuid import uuid4

from loguru import logger
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.models import GeneratedResponse, Query, RetrievedChunk
from src.observability.metrics import (
    generation_requests_total,
    generation_tokens_total,
    generation_duration,
    generation_confidence,
    generation_cost,
)


class ResponseGenerator:
    """LLM-based response generation with citations."""

    # OpenAI pricing (as of 2025)
    PRICING = {
        "gpt-4-turbo-preview": {"input": 0.01 / 1000, "output": 0.03 / 1000},
        "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000},
        "gpt-3.5-turbo": {"input": 0.0005 / 1000, "output": 0.0015 / 1000},
    }

    def __init__(self):
        """Initialize response generator."""
        self.settings = get_settings()
        self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        self.model = self.settings.openai_llm_model
        
        logger.info(f"ResponseGenerator initialized with model: {self.model}")

    async def generate(
        self,
        query: Query,
        retrieved_chunks: List[RetrievedChunk]
    ) -> GeneratedResponse:
        """Generate response with citations.

        Args:
            query: User query
            retrieved_chunks: Retrieved context chunks

        Returns:
            Generated response with citations
        """
        logger.info(f"Generating response for query: {query.text[:100]}")
        
        start_time = time.time()

        # Compress context if needed
        context = self._build_context(retrieved_chunks)
        
        # Build prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query.text, context)

        # Generate response
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.settings.generation_temperature,
                max_tokens=self.settings.generation_max_tokens,
            )

            # Extract response
            response_text = completion.choices[0].message.content
            
            # Parse response (expecting JSON format)
            try:
                response_data = json.loads(response_text)
                answer = response_data.get("answer", "")
                citations = response_data.get("citations", [])
                confidence = response_data.get("confidence", 0.5)
            except json.JSONDecodeError:
                # Fallback if not JSON
                logger.warning("Response not in JSON format, using raw text")
                answer = response_text
                citations = []
                confidence = 0.5

            # Calculate tokens
            tokens_used = completion.usage.total_tokens
            input_tokens = completion.usage.prompt_tokens
            output_tokens = completion.usage.completion_tokens

            duration = (time.time() - start_time) * 1000  # ms

            # Calculate cost
            cost = self._calculate_cost(input_tokens, output_tokens)

            # Check confidence threshold
            if confidence < self.settings.generation_confidence_threshold:
                logger.warning(
                    f"Response confidence {confidence:.2f} below threshold "
                    f"{self.settings.generation_confidence_threshold}"
                )
                if self.settings.generation_require_citation:
                    answer = (
                        "I don't have enough confidence to answer this question "
                        "based on the available documents. Please provide more "
                        "specific information or rephrase your question."
                    )
                    confidence = 0.0

            # Create response object
            generated_response = GeneratedResponse(
                query_id=query.id,
                answer=answer,
                citations=citations,
                retrieved_chunks=retrieved_chunks,
                confidence=confidence,
                tokens_used=tokens_used,
                latency_ms=duration,
                model=self.model,
                metadata={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": cost
                }
            )

            # Update metrics
            generation_requests_total.labels(
                model=self.model, tenant_id=query.tenant_id
            ).inc()
            generation_tokens_total.labels(model=self.model, type="input").inc(
                input_tokens
            )
            generation_tokens_total.labels(model=self.model, type="output").inc(
                output_tokens
            )
            generation_duration.labels(model=self.model).observe(duration / 1000)
            generation_confidence.labels(tenant_id=query.tenant_id).observe(confidence)
            generation_cost.labels(model=self.model).observe(cost)

            logger.info(
                f"Response generated: {tokens_used} tokens, "
                f"{duration:.0f}ms, confidence={confidence:.2f}, ${cost:.6f}"
            )

            return generated_response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def _build_system_prompt(self) -> str:
        """Build system prompt with instructions.

        Returns:
            System prompt
        """
        return """You are an enterprise compliance and policy intelligence assistant.

Your task is to answer questions based ONLY on the provided context documents.

CRITICAL RULES:
1. ONLY use information from the provided context
2. ALWAYS cite your sources using [Source X] notation
3. If the answer is not in the context, say "I don't have enough information"
4. Never make up or hallucinate information
5. Provide confidence score (0.0-1.0) based on context quality
6. Be precise and professional

OUTPUT FORMAT (JSON):
{
    "answer": "Your detailed answer with [Source 1], [Source 2] citations",
    "citations": [
        {
            "source_id": 1,
            "document": "filename.pdf",
            "chunk_text": "Relevant excerpt from document",
            "relevance": "Why this source is relevant"
        }
    ],
    "confidence": 0.95
}

If you cannot answer confidently (confidence < 0.65), set confidence to 0.0 and explain why in the answer."""

    def _build_user_prompt(self, query: str, context: str) -> str:
        """Build user prompt with query and context.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            User prompt
        """
        return f"""CONTEXT:
{context}

QUESTION:
{query}

Please provide your answer in JSON format as specified."""

    def _build_context(self, retrieved_chunks: List[RetrievedChunk]) -> str:
        """Build context string from retrieved chunks.

        Args:
            retrieved_chunks: Retrieved chunks

        Returns:
            Context string
        """
        if not retrieved_chunks:
            return "No relevant context found."

        context_parts = []
        for i, retrieved in enumerate(retrieved_chunks, 1):
            chunk = retrieved.chunk
            doc_name = chunk.metadata.get("document_filename", "Unknown")
            
            context_parts.append(
                f"[Source {i}] (Document: {doc_name}, Score: {retrieved.score:.3f})\n"
                f"{chunk.content}\n"
            )

        return "\n---\n".join(context_parts)

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate generation cost.

        Args:
            input_tokens: Input tokens
            output_tokens: Output tokens

        Returns:
            Cost in USD
        """
        pricing = self.PRICING.get(self.model, {"input": 0.01/1000, "output": 0.03/1000})
        
        input_cost = input_tokens * pricing["input"]
        output_cost = output_tokens * pricing["output"]
        
        return input_cost + output_cost


class HallucinationDetector:
    """Detect hallucinations in generated responses."""

    def __init__(self):
        """Initialize hallucination detector."""
        self.settings = get_settings()

    def detect(
        self,
        response: GeneratedResponse,
        query_text: str
    ) -> Dict[str, Any]:
        """Detect potential hallucinations.

        Args:
            response: Generated response
            query_text: Original query

        Returns:
            Detection results
        """
        logger.info("Running hallucination detection")

        issues = []
        hallucination_score = 0.0

        # Check 1: Answer requires citations
        if self.settings.generation_require_citation:
            if not response.citations:
                issues.append("No citations provided")
                hallucination_score += 0.5

        # Check 2: Citations match retrieved chunks
        cited_sources = set()
        for citation in response.citations:
            source_id = citation.get("source_id")
            if source_id:
                cited_sources.add(source_id)

        if cited_sources and len(cited_sources) > len(response.retrieved_chunks):
            issues.append("More citations than retrieved chunks")
            hallucination_score += 0.3

        # Check 3: Low confidence
        if response.confidence < self.settings.generation_confidence_threshold:
            issues.append(f"Low confidence: {response.confidence:.2f}")
            hallucination_score += 0.2

        # Check 4: Answer contains "I don't know" patterns
        uncertain_patterns = [
            "i don't have",
            "i cannot",
            "not sure",
            "unclear",
            "no information"
        ]
        answer_lower = response.answer.lower()
        if any(pattern in answer_lower for pattern in uncertain_patterns):
            # This is actually good - the model is being honest
            hallucination_score = max(0.0, hallucination_score - 0.3)

        # Normalize score
        hallucination_score = min(1.0, max(0.0, hallucination_score))

        is_hallucinated = hallucination_score > self.settings.eval_max_hallucination_rate

        result = {
            "is_hallucinated": is_hallucinated,
            "hallucination_score": hallucination_score,
            "issues": issues,
            "passed": not is_hallucinated
        }

        if is_hallucinated:
            logger.warning(
                f"Potential hallucination detected (score: {hallucination_score:.2f})"
            )

        return result
