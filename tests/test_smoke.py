"""Smoke tests — verify imports and basic config loading."""

from __future__ import annotations

import os

import pytest


def test_imports() -> None:
    """Every module imports cleanly."""
    import compliance_rag  # noqa: F401
    import compliance_rag.agent  # noqa: F401
    import compliance_rag.cli  # noqa: F401
    import compliance_rag.corpora  # noqa: F401
    import compliance_rag.eval  # noqa: F401
    import compliance_rag.ingest  # noqa: F401
    import compliance_rag.retrieve  # noqa: F401
    import compliance_rag.schemas  # noqa: F401


def test_criterion_id_pattern_nist() -> None:
    """Regex pulls NIST control IDs from natural-language queries."""
    from compliance_rag.retrieve import CRITERION_ID_PATTERN

    cases = {
        "What evidence supports AC-2?": ["AC-2"],
        "Compare AC-3 and AC-6(1).": ["AC-3", "AC-6(1)"],
        "Tell me about access control.": [],
    }
    for query, expected in cases.items():
        assert CRITERION_ID_PATTERN.findall(query) == expected


def test_criterion_id_pattern_aicpa() -> None:
    """Regex pulls AICPA TSC criterion IDs."""
    from compliance_rag.retrieve import CRITERION_ID_PATTERN

    assert CRITERION_ID_PATTERN.findall("Does CC6.1 cover MFA?") == ["CC6.1"]
    assert CRITERION_ID_PATTERN.findall("A1.2 vs PI1.1") == ["A1.2", "PI1.1"]


def test_nist_chunker_loads() -> None:
    """The NIST chunker loads chunks from the on-disk CSV (must be present)."""
    from compliance_rag.corpora import get_corpus

    corpus = get_corpus("nist-800-53")
    if not corpus.is_available():
        pytest.skip("NIST CSV not on disk; skipping")

    chunks = list(corpus.load_chunks())
    assert len(chunks) > 500, "expected hundreds of chunks from NIST 800-53 r5"
    # AC-2 should be present and have an enhancement.
    by_id = {c.criterion_id: c for c in chunks}
    assert "AC-2" in by_id
    assert "AC-2(1)" in by_id
    assert by_id["AC-2"].is_root
    assert by_id["AC-2(1)"].parent_id == "AC-2"
    assert "Account Management" in by_id["AC-2"].title


def test_corpus_lookup() -> None:
    """Both corpora can be instantiated."""
    from compliance_rag.corpora import get_corpus

    nist = get_corpus("nist-800-53")
    assert nist.name == "nist-800-53"

    tsc = get_corpus("aicpa-tsc")
    assert tsc.name == "aicpa-tsc"

    with pytest.raises(ValueError):
        get_corpus("nonexistent")


def test_config_requires_api_keys() -> None:
    """Config.from_env() fails clearly when keys are missing."""
    from compliance_rag.config import Config

    old_anthropic = os.environ.pop("ANTHROPIC_API_KEY", None)
    old_openai = os.environ.pop("OPENAI_API_KEY", None)
    try:
        with pytest.raises(RuntimeError, match="Missing required env var"):
            Config.from_env()
    finally:
        if old_anthropic is not None:
            os.environ["ANTHROPIC_API_KEY"] = old_anthropic
        if old_openai is not None:
            os.environ["OPENAI_API_KEY"] = old_openai
