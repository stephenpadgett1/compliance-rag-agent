"""PDF / CSV → chunks → embeddings → Chroma.

Design notes:

- **Embeddings are computed once and persisted.** The agent runtime reads from
  Chroma only — no embedding calls in the hot path.

- **OpenAI text-embedding-3-small.** 1536 dims, $0.02/1M tokens, fast. Easy to
  swap behind the `embed` helper if you want Voyage or local models later.

- **Batch the embedding calls.** OpenAI accepts up to 2048 inputs per request;
  we batch to keep one round-trip per ~100 chunks.

- **Idempotent.** Re-running clears the Chroma collection and rebuilds. Source
  documents change so rarely that incremental updates aren't worth the code.
"""

from __future__ import annotations

from collections.abc import Iterable

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from compliance_rag.config import Config
from compliance_rag.corpora import get_corpus
from compliance_rag.corpora.base import CorpusChunk

console = Console()

COLLECTION_PREFIX = "compliance_rag_"
EMBED_BATCH_SIZE = 100


def collection_name(corpus_name: str) -> str:
    """Chroma collection name for a corpus — keeps multiple corpora isolated."""
    return COLLECTION_PREFIX + corpus_name.replace("-", "_")


def embed_batch(client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
    """One embedding round-trip. Returns vectors in the same order as `texts`."""
    response = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in response.data]


def embed_all(
    client: OpenAI, model: str, chunks: list[CorpusChunk]
) -> list[list[float]]:
    """Embed every chunk, batching to keep API round-trips reasonable."""
    vectors: list[list[float]] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding", total=len(chunks))
        for i in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[i : i + EMBED_BATCH_SIZE]
            vectors.extend(embed_batch(client, model, [c.text for c in batch]))
            progress.update(task, advance=len(batch))
    return vectors


def run_ingest(config: Config) -> int:
    """Top-level ingest pipeline. Returns the number of chunks indexed."""
    corpus = get_corpus(config.corpus)
    if not corpus.is_available():
        raise FileNotFoundError(
            f"Source not found at {corpus.expected_source_path()}. "
            f"{corpus.distribution_note}"
        )

    console.print(f"[bold]Loading chunks from[/bold] {corpus.display_name}...")
    chunks: list[CorpusChunk] = list(corpus.load_chunks())
    if not chunks:
        raise RuntimeError(
            f"Loaded 0 chunks from {corpus.name}. Check the source file is intact."
        )
    console.print(f"  → {len(chunks)} chunks")

    openai_client = OpenAI(api_key=config.openai_api_key)
    vectors = embed_all(openai_client, config.embedding_model, chunks)

    config.chroma_path.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(
        path=str(config.chroma_path),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )

    name = collection_name(corpus.name)
    # Clean rebuild — see module docstring on idempotency.
    try:
        chroma_client.delete_collection(name)
    except Exception:  # noqa: BLE001 — Chroma's "not found" raises a generic exception
        pass

    collection = chroma_client.create_collection(name=name, metadata={"corpus": corpus.name})

    metadata = [_chunk_metadata(c) for c in chunks]
    collection.add(
        ids=[c.criterion_id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=vectors,
        metadatas=metadata,
    )

    console.print(
        f"[green]✓[/green] Indexed {len(chunks)} chunks into collection [bold]{name}[/bold]"
    )
    console.print(f"  Persisted to {config.chroma_path}")
    return len(chunks)


def _chunk_metadata(chunk: CorpusChunk) -> dict[str, str | int | bool]:
    """Flatten a CorpusChunk's metadata for Chroma (only str/int/float/bool allowed)."""
    meta: dict[str, str | int | bool] = {
        "criterion_id": chunk.criterion_id,
        "title": chunk.title,
        "is_root": chunk.is_root,
    }
    if chunk.parent_id is not None:
        meta["parent_id"] = chunk.parent_id
    if chunk.page_number is not None:
        meta["page_number"] = chunk.page_number
    # Allow corpus-specific extras (family, related, etc.) — flatten to scalars.
    for k, v in chunk.extra.items():
        if isinstance(v, (str, int, float, bool)):
            meta[k] = v
    return meta


def first_n_chunks(chunks: Iterable[CorpusChunk], n: int) -> list[CorpusChunk]:
    """Helper for debugging — peek at first N chunks without consuming the iterator."""
    out: list[CorpusChunk] = []
    for c in chunks:
        out.append(c)
        if len(out) >= n:
            break
    return out
