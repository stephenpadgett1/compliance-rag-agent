"""Centralized config — env vars and model IDs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    anthropic_api_key: str
    openai_api_key: str

    corpus: str
    chroma_path: Path
    embedding_model: str

    agent_model: str
    judge_model: str

    @classmethod
    def from_env(cls) -> Config:
        try:
            anthropic_key = os.environ["ANTHROPIC_API_KEY"]
            openai_key = os.environ["OPENAI_API_KEY"]
        except KeyError as e:
            raise RuntimeError(
                f"Missing required env var: {e.args[0]}. "
                "Copy .env.example to .env and fill in the keys."
            ) from None

        chroma_path = Path(os.environ.get("CHROMA_PATH", "data/chroma_db"))
        if not chroma_path.is_absolute():
            chroma_path = REPO_ROOT / chroma_path

        return cls(
            anthropic_api_key=anthropic_key,
            openai_api_key=openai_key,
            corpus=os.environ.get("CORPUS", "nist-800-53"),
            chroma_path=chroma_path,
            embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
            agent_model=os.environ.get("AGENT_MODEL", "claude-sonnet-4-6"),
            judge_model=os.environ.get("JUDGE_MODEL", "claude-opus-4-7"),
        )
