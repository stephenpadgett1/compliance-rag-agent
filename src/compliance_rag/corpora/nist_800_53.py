"""NIST SP 800-53 Rev 5 — public default corpus.

Ingests from the official CSV catalog published by NIST. Each row is one control
or enhancement; we emit one CorpusChunk per non-withdrawn row.

CSV columns: identifier, name, control_text, discussion, related, (trailing empty)
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterator
from pathlib import Path

from compliance_rag.config import REPO_ROOT
from compliance_rag.corpora.base import Corpus, CorpusChunk

ENHANCEMENT_SUFFIX = re.compile(r"\(\d+\)$")
WITHDRAWN_PREFIX = "Withdrawn:"


class NIST80053Corpus(Corpus):
    name = "nist-800-53"
    display_name = "NIST SP 800-53 Rev 5"
    source_url = (
        "https://csrc.nist.gov/CSRC/media/Projects/risk-management/"
        "800-53%20Downloads/800-53r5/NIST_SP-800-53_rev5_catalog_load.csv"
    )
    distribution_note = "Public domain (NIST publication). Run `compliance-rag download-corpus`."

    def expected_source_path(self) -> Path:
        return REPO_ROOT / "data" / "nist-800-53-r5.csv"

    def is_available(self) -> bool:
        return self.expected_source_path().exists()

    def load_chunks(self) -> Iterator[CorpusChunk]:
        path = self.expected_source_path()
        if not path.exists():
            raise FileNotFoundError(
                f"NIST 800-53 CSV not found at {path}. Run `compliance-rag download-corpus`."
            )

        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                chunk = self._row_to_chunk(row)
                if chunk is not None:
                    yield chunk

    @staticmethod
    def _row_to_chunk(row: dict[str, str]) -> CorpusChunk | None:
        identifier = (row.get("identifier") or "").strip()
        name = (row.get("name") or "").strip()
        control_text = (row.get("control_text") or "").strip()
        discussion = (row.get("discussion") or "").strip()
        related = (row.get("related") or "").strip()

        if not identifier or not control_text:
            return None

        # Skip withdrawn controls — they're pointers to other controls, not substantive
        if control_text.lstrip().startswith(WITHDRAWN_PREFIX):
            return None

        parent_id: str | None = None
        if ENHANCEMENT_SUFFIX.search(identifier):
            parent_id = ENHANCEMENT_SUFFIX.sub("", identifier)

        # Compose the chunk text. Keep section headers so the embedding picks up
        # the semantic structure ("Discussion:", "Related:") not just the prose.
        parts = [f"{identifier} {name}", "", "Control:", control_text]
        if discussion:
            parts += ["", "Discussion:", discussion]
        if related and related != "[None]":
            parts += ["", "Related Controls:", related]
        text = "\n".join(parts)

        return CorpusChunk(
            criterion_id=identifier,
            title=name,
            text=text,
            parent_id=parent_id,
            extra={"related": related, "family": identifier.split("-")[0]},
        )
