"""PII detection and redaction."""

import re
from typing import Dict, List, Tuple

from loguru import logger

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available, using regex-only PII detection")

from src.config import get_settings
from src.observability.metrics import pii_detections_total


class PIIDetector:
    """PII detection using regex and NER."""

    # PII patterns
    PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b",
        "ssn": r"\b(?!000|666)[0-8][0-9]{2}-(?!00)[0-9]{2}-(?!0000)[0-9]{4}\b",
        "credit_card": r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|(?:2131|1800|35\d{3})\d{11})\b",
        "ip_address": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
    }

    def __init__(self):
        """Initialize PII detector."""
        self.settings = get_settings()
        self.nlp = None
        
        # Load spaCy model if available
        if SPACY_AVAILABLE and self.settings.enable_pii_redaction:
            try:
                self.nlp = spacy.load(self.settings.pii_detection_model)
                logger.info(f"Loaded spaCy model: {self.settings.pii_detection_model}")
            except Exception as e:
                logger.warning(f"Could not load spaCy model: {e}")
                self.nlp = None

    def detect(self, text: str) -> List[Dict[str, any]]:
        """Detect PII in text.

        Args:
            text: Input text

        Returns:
            List of detected PII entities
        """
        if not self.settings.enable_pii_redaction:
            return []

        detections = []

        # Regex-based detection
        for pii_type, pattern in self.PATTERNS.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                detections.append({
                    "type": pii_type,
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "method": "regex"
                })

        # NER-based detection
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "GPE", "ORG", "DATE", "MONEY"]:
                    # Only flag PERSON as PII
                    if ent.label_ == "PERSON":
                        detections.append({
                            "type": "person_name",
                            "value": ent.text,
                            "start": ent.start_char,
                            "end": ent.end_char,
                            "method": "ner",
                            "confidence": 0.9  # spaCy doesn't provide scores
                        })

        return detections

    def redact(self, text: str, tenant_id: str = "default") -> Tuple[str, List[Dict]]:
        """Detect and redact PII from text.

        Args:
            text: Input text
            tenant_id: Tenant ID for metrics

        Returns:
            Tuple of (redacted_text, detections)
        """
        detections = self.detect(text)

        if not detections:
            return text, []

        logger.info(f"Found {len(detections)} PII entities")

        # Sort by position (reverse order to maintain indices)
        detections.sort(key=lambda x: x["start"], reverse=True)

        # Redact
        redacted_text = text
        for detection in detections:
            start = detection["start"]
            end = detection["end"]
            pii_type = detection["type"]
            
            # Replace with placeholder
            placeholder = f"[REDACTED_{pii_type.upper()}]"
            redacted_text = redacted_text[:start] + placeholder + redacted_text[end:]

            # Update metrics
            pii_detections_total.labels(
                pii_type=pii_type, tenant_id=tenant_id
            ).inc()

        return redacted_text, detections

    def contains_pii(self, text: str) -> bool:
        """Check if text contains PII.

        Args:
            text: Input text

        Returns:
            True if PII detected
        """
        detections = self.detect(text)
        return len(detections) > 0