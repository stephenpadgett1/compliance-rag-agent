"""Corpus base interface.

A Corpus encapsulates everything that varies between compliance frameworks:
- where to find the source data
- how to parse it into criterion-aligned chunks
- how citations should be formatted

NIST 800-53 ingests from the official CSV; AICPA TSC ingests from a user-supplied PDF.
Both produce the same `CorpusChunk` shape so downstream code is corpus-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CorpusChunk:
    """One semantically-bounded chunk of source material.

    Boundaries follow the corpus's natural structure (control / criterion / point of focus),
    not fixed token counts.
    """

    criterion_id: str
    title: str
    text: str
    parent_id: str | None = None  # e.g. AC-2(1) is a child of AC-2
    page_number: int | None = None  # populated when source is PDF
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def is_root(self) -> bool:
        """True if this is a top-level control (no parent enhancement)."""
        return self.parent_id is None


class Corpus(ABC):
    """A compliance-framework corpus profile."""

    name: str
    display_name: str
    source_url: str | None
    distribution_note: str | None

    @abstractmethod
    def expected_source_path(self) -> Path:
        """Where the source data is expected to live on disk."""

    @abstractmethod
    def is_available(self) -> bool:
        """True if the source data is on disk and ready to ingest."""

    @abstractmethod
    def load_chunks(self) -> Iterator[CorpusChunk]:
        """Parse the source into criterion-boundary-aligned chunks."""
