"""AICPA 2017 Trust Services Criteria — optional corpus, user-supplied PDF.

The PDF is freely downloadable from the AICPA but gated behind a free account
registration, so we cannot redistribute it. Users supply their own copy.

Unlike NIST, AICPA does not publish a machine-readable catalog — so this path
relies on PDF parsing.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from compliance_rag.config import REPO_ROOT
from compliance_rag.corpora.base import Corpus, CorpusChunk


class AICPATSCCorpus(Corpus):
    name = "aicpa-tsc"
    display_name = "AICPA 2017 Trust Services Criteria (with 2022 Revised Points of Focus)"
    source_url = (
        "https://www.aicpa-cima.com/resources/download/"
        "2017-trust-services-criteria-with-revised-points-of-focus-2022"
    )
    distribution_note = (
        "Requires a free AICPA account registration; not redistributable. "
        "Download manually and place at data/aicpa-tsc-2017.pdf"
    )

    def expected_source_path(self) -> Path:
        return REPO_ROOT / "data" / "aicpa-tsc-2017.pdf"

    def is_available(self) -> bool:
        return self.expected_source_path().exists()

    def load_chunks(self) -> Iterator[CorpusChunk]:
        """Parse the TSC PDF into per-criterion chunks.

        TSC structure:
        - Five categories: Security, Availability, Processing Integrity,
          Confidentiality, Privacy
        - Common Criteria (CC1.0 - CC9.0), each with sub-criteria (CC6.1, CC6.2, ...)
        - Each criterion has a Statement and Points of Focus (bulleted guidance)

        TODO: implement once a user supplies the AICPA TSC PDF. The pypdf-based
        parser will look for criterion-ID patterns (CC\\d+\\.\\d+, A\\d+\\.\\d+, etc.),
        extract the statement and Points of Focus per criterion, and emit one
        CorpusChunk per criterion.
        """
        raise NotImplementedError(
            "AICPA TSC ingest is implemented behind a user-supplied PDF. "
            "Download the TSC and place at data/aicpa-tsc-2017.pdf, then this "
            "method will parse it. See ROADMAP.md."
        )
