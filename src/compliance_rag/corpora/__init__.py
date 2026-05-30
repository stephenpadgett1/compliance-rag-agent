"""Corpus profiles.

A corpus profile encapsulates everything that varies between compliance frameworks:
- where to find the source PDF
- how to parse it into chunks (criterion boundaries differ per framework)
- how citations should be formatted (CC6.1 vs AC-2)
"""

from __future__ import annotations

from compliance_rag.corpora.base import Corpus, CorpusChunk

__all__ = ["Corpus", "CorpusChunk", "get_corpus"]


def get_corpus(name: str) -> Corpus:
    """Look up a corpus profile by name."""
    if name == "nist-800-53":
        from compliance_rag.corpora.nist_800_53 import NIST80053Corpus

        return NIST80053Corpus()
    if name == "aicpa-tsc":
        from compliance_rag.corpora.aicpa_tsc import AICPATSCCorpus

        return AICPATSCCorpus()
    raise ValueError(
        f"Unknown corpus: {name!r}. Supported: 'nist-800-53', 'aicpa-tsc'."
    )
