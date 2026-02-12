"""Golden Q&A dataset management for evaluation."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, Field

from src.config import get_settings


class GoldenQA(BaseModel):
    """Golden Q&A pair for evaluation."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    expected_answer: Optional[str] = None
    relevant_chunk_ids: List[str] = Field(default_factory=list)
    relevant_document_ids: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    difficulty: Optional[str] = None  # "easy", "medium", "hard"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class GoldenDataset(BaseModel):
    """Collection of golden Q&A pairs."""

    name: str
    description: Optional[str] = None
    version: str = "1.0"
    qa_pairs: List[GoldenQA] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GoldenDatasetManager:
    """Manage golden Q&A datasets for evaluation."""

    def __init__(self):
        """Initialize dataset manager."""
        self.settings = get_settings()
        self.dataset_path = Path(self.settings.eval_dataset_path)
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)

    def create_dataset(
        self,
        name: str,
        description: Optional[str] = None
    ) -> GoldenDataset:
        """Create new golden dataset.

        Args:
            name: Dataset name
            description: Dataset description

        Returns:
            Golden dataset
        """
        dataset = GoldenDataset(
            name=name,
            description=description
        )
        logger.info(f"Created golden dataset: {name}")
        return dataset

    def add_qa_pair(
        self,
        dataset: GoldenDataset,
        question: str,
        expected_answer: Optional[str] = None,
        relevant_chunk_ids: Optional[List[str]] = None,
        relevant_document_ids: Optional[List[str]] = None,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        **metadata
    ) -> GoldenQA:
        """Add Q&A pair to dataset.

        Args:
            dataset: Golden dataset
            question: Question text
            expected_answer: Expected answer (optional)
            relevant_chunk_ids: List of relevant chunk IDs
            relevant_document_ids: List of relevant document IDs
            category: Question category
            difficulty: Question difficulty
            **metadata: Additional metadata

        Returns:
            Golden Q&A pair
        """
        qa = GoldenQA(
            question=question,
            expected_answer=expected_answer,
            relevant_chunk_ids=relevant_chunk_ids or [],
            relevant_document_ids=relevant_document_ids or [],
            category=category,
            difficulty=difficulty,
            metadata=metadata
        )

        dataset.qa_pairs.append(qa)
        dataset.updated_at = datetime.utcnow()

        logger.info(f"Added Q&A pair to dataset '{dataset.name}': {question[:50]}")

        return qa

    def save_dataset(self, dataset: GoldenDataset, path: Optional[Path] = None) -> None:
        """Save dataset to file.

        Args:
            dataset: Golden dataset
            path: Save path (default: from config)
        """
        if path is None:
            path = self.dataset_path

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                dataset.model_dump(mode="json"),
                f,
                indent=2,
                default=str
            )

        logger.info(f"Saved dataset '{dataset.name}' to {path}")

    def load_dataset(self, path: Optional[Path] = None) -> GoldenDataset:
        """Load dataset from file.

        Args:
            path: Load path (default: from config)

        Returns:
            Golden dataset
        """
        if path is None:
            path = self.dataset_path

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset = GoldenDataset(**data)

        logger.info(
            f"Loaded dataset '{dataset.name}' with {len(dataset.qa_pairs)} Q&A pairs"
        )

        return dataset

    def get_by_category(self, dataset: GoldenDataset, category: str) -> List[GoldenQA]:
        """Get Q&A pairs by category.

        Args:
            dataset: Golden dataset
            category: Category name

        Returns:
            List of Q&A pairs
        """
        return [qa for qa in dataset.qa_pairs if qa.category == category]

    def get_by_difficulty(self, dataset: GoldenDataset, difficulty: str) -> List[GoldenQA]:
        """Get Q&A pairs by difficulty.

        Args:
            dataset: Golden dataset
            difficulty: Difficulty level

        Returns:
            List of Q&A pairs
        """
        return [qa for qa in dataset.qa_pairs if qa.difficulty == difficulty]

    def create_sample_dataset(self) -> GoldenDataset:
        """Create sample dataset for testing.

        Returns:
            Sample golden dataset
        """
        dataset = self.create_dataset(
            name="compliance_sample",
            description="Sample Q&A pairs for compliance policy testing"
        )

        # Sample questions
        sample_questions = [
            {
                "question": "What is the data retention policy for customer records?",
                "category": "data_retention",
                "difficulty": "easy"
            },
            {
                "question": "What are the requirements for password complexity?",
                "category": "security",
                "difficulty": "easy"
            },
            {
                "question": "What is the process for handling data breach incidents?",
                "category": "incident_response",
                "difficulty": "medium"
            },
            {
                "question": "What are the GDPR compliance requirements for international data transfers?",
                "category": "gdpr",
                "difficulty": "hard"
            },
            {
                "question": "What is the escalation procedure for compliance violations?",
                "category": "violations",
                "difficulty": "medium"
            }
        ]

        for q in sample_questions:
            self.add_qa_pair(dataset, **q)

        logger.info(f"Created sample dataset with {len(sample_questions)} questions")

        return dataset