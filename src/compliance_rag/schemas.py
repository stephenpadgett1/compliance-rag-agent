"""Pydantic models for the public API surface.

Two principles:
1. Citations are first-class. Every answer carries an array of structured citations.
2. Eval cases declare the rubric upfront, so the eval harness is auditable.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    """The four question shapes the agent is graded against."""

    CONTROL_MAPPING = "control_mapping"
    CRITERIA_INTERPRETATION = "criteria_interpretation"
    OPERATIONAL = "operational"
    GAP_ANALYSIS = "gap_analysis"


class Citation(BaseModel):
    """A pointer back into the source corpus for a claim in the answer."""

    criterion_id: str = Field(
        ..., description="The criterion or control ID — e.g. 'AC-2' (NIST) or 'CC6.1' (TSC)."
    )
    excerpt: str = Field(
        ..., description="A short verbatim quote from the source supporting the claim."
    )
    page_number: int | None = Field(
        default=None, description="Source PDF page number, if known."
    )


class Answer(BaseModel):
    """The agent's output contract."""

    question: str
    answer: str = Field(..., description="The grounded answer in plain English.")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Citations supporting each load-bearing claim in `answer`.",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="The agent's self-rated confidence after retrieval and critique.",
    )


class EvalCase(BaseModel):
    """One row in the eval fixture set."""

    id: str
    question: str
    question_type: QuestionType
    expected_criteria: list[str] = Field(
        default_factory=list,
        description="Criteria/controls the answer should cite. Used by the judge.",
    )
    notes: str | None = Field(
        default=None, description="Optional grader hints, e.g. expected emphasis."
    )


class JudgeVerdict(BaseModel):
    """LLM-as-judge output for a single eval case."""

    case_id: str
    faithfulness: int = Field(..., ge=0, le=5, description="0–5: are claims grounded in citations?")
    citation_accuracy: int = Field(
        ..., ge=0, le=5, description="0–5: do citations actually support the claims?"
    )
    usefulness: int = Field(
        ..., ge=0, le=5, description="0–5: useful answer vs evasive/hedged?"
    )
    notes: str = Field(default="", description="Judge's reasoning, max 2-3 sentences.")
