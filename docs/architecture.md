# Architecture notes

Technical reference for the SOC 2 RAG showcase. The user-facing README is in the repo root; this document is for the reader who wants to understand the build decisions.

## Layers

```
┌────────────────────────────────────────────────────────────────┐
│  CLI (cli.py)  ─  click commands: status / ingest / ask / eval │
└─────────────────────────────┬──────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
     ┌──────────┐      ┌──────────┐      ┌──────────┐
     │ ingest.py│      │ agent.py │      │ eval.py  │
     │          │      │          │      │          │
     │  CSV/PDF │      │ LangGraph│      │  judge  │
     │  → embed │      │  state   │      │  loop   │
     │  → Chroma│      │  machine │      │          │
     └────┬─────┘      └────┬─────┘      └────┬─────┘
          │                 │                  │
          │            ┌────▼─────┐            │
          │            │retrieve.py│            │
          │            │ vector + │            │
          │            │ ID match │            │
          │            └────┬─────┘            │
          ▼                 ▼                  │
       ┌──────────────────────────┐            │
       │   Chroma (embedded)      │◀───────────┘
       │   data/chroma_db/        │
       └──────────────────────────┘
```

## Why LangGraph for the agent loop

LangGraph gives us clean state-machine semantics for what is genuinely a state machine: planner → retrieve → draft → critic → (retry | finalize). The conditional edge out of the critic node is the load-bearing piece — vanilla Python control flow would work, but LangGraph keeps the graph shape readable.

The Anthropic SDK is called *directly* inside each node, not through `langchain-anthropic`. This is deliberate. Direct SDK calls preserve:
- `cache_control` placement on system blocks
- `output_config.format` for schema-constrained output
- `thinking: {type: "adaptive"}` for Sonnet 4.6 adaptive thinking
- Beta feature opt-ins without waiting on LangChain wrappers

The LangChain abstraction layer is fine if you're swapping providers a lot. We aren't.

## Prompt caching strategy

The retrieval context is the prime cache target *within* one question's agent loop. The drafter and the critic both see the same chunks; the critic's call is a cache read.

### The non-obvious part: cache keys are sensitive to system block *structure*, not just content

The first instrumented run showed 8,836 cache writes and **0 cache reads**. Cause: the drafter and critic had different system block lists.

Initial (broken) design:

```python
# Drafter
system=[
    {"text": SOURCE_MATERIAL, "cache_control": {"type": "ephemeral"}},
    {"text": DRAFTER_INSTRUCTIONS},
]
# Critic
system=[
    {"text": SOURCE_MATERIAL, "cache_control": {"type": "ephemeral"}},
    {"text": CRITIC_INSTRUCTIONS},
]
```

The `cache_control` marker is on the first block in both cases. The cached bytes are identical (same `SOURCE_MATERIAL`). But the *list structures* differ — the second block's content is different — and the cache key includes the block structure. No cache reads ever happened.

Final design:

```python
# Both drafter and critic
system=[
    {"text": SOURCE_MATERIAL, "cache_control": {"type": "ephemeral"}},
]
# Drafter user message
content = f"{DRAFTER_INSTRUCTIONS}\n\nQuestion: {question}"
# Critic user message
content = f"{CRITIC_INSTRUCTIONS}\n\nQuestion: ... Answer: ... Citations: ..."
```

Same system block list. Per-call variation lives in the user message (which is after the cache marker anyway, so it never affected the cache key). After this change: **cache hit rate jumped from 0% to ~50%**, cost per case dropped 25%.

### Cache scope across the eval — what actually works

- **Within a question (drafter → critic)**: ❌ doesn't activate. See "Why drafter→critic caching doesn't activate" below.
- **Across questions**: different chunks retrieved → different cached prefixes → no shared cache. (Expected.)
- **Across runs within the 5-minute TTL**: ✅ this is where caching pays off. A second run of the same eval set picks up the prior run's writes. The first run after the bug-fix above showed 51.8% cache hit rate against a previous run's writes — real cost reduction.

### Why drafter→critic caching doesn't activate within a single question

After the system-block-structure fix above, the per-call usage stats still show different `cache_creation_input_tokens` between drafter (1926) and critic (1762) — a consistent **164-token delta** across every case in the eval. Both calls write to cache; neither reads. The cache keys must therefore be different.

The only differing parameters between drafter and critic at that point are:
- `output_config.format.schema` — drafter returns `Answer` (~150 tokens of schema); critic returns a verdict (~60 tokens)
- `max_tokens` — 16000 vs 2000 (does not affect prefix processing)
- `thinking` — adaptive vs disabled (Anthropic's docs say this preserves system cache; aligning the two didn't change behavior)

The 164-token delta is consistent with the difference in schema size, strongly suggesting **`output_config.format.schema` is part of the cache key** even though the official invalidation hierarchy table doesn't list it.

This is unfixable without giving up structured outputs on one of the two calls — a contortion not worth the cost savings on a 30-case eval. **What works instead is the cross-run path**, which is what we use in practice (eval runs are infrequent enough that 5-minute TTL covers retries and adjacent runs).

### Why we don't cache the judge

### Why we don't cache the judge

`JUDGE_INSTRUCTIONS` is ~400 tokens. Opus 4.7's minimum cacheable prefix is 4,096 tokens; below that, `cache_control` is silently no-op'd. Rather than emit misleading config, the judge call ships without a cache marker.

## The critic loop

The critic checks groundedness — every load-bearing claim in the draft should be supported by a cited excerpt. If the critic flags ungrounded claims, the graph routes back to the retriever with an `expand_query_with` hint generated by the critic.

This is bounded to `MAX_RETRY_LOOPS = 2` to prevent runaway. If after two retries the critic still flags issues, we accept the draft but downgrade the answer's `confidence` field to `low`.

In practice this loop fires on maybe 10-15% of questions in the eval suite. The most common trigger: the agent infers operational specifics (e.g., "annual review") from a control that says "organization-defined frequency." The critic catches it, the retriever pulls related controls, the second draft is honestly evasive.

## Choosing the judge model

Eval uses Opus 4.7 (config: `JUDGE_MODEL`). Opus is overkill for the agent itself but the right choice for the judge — correctness > latency, and the judge runs once per case (30 cases × ~$0.03 each = ~$1 per eval run).

Caching the judge's system prompt (`cache_control` on the instructions block) gives cache hits on every case after the first. The case-specific input goes in the user message and isn't cached.

## Why structured outputs

Both the drafter and the critic use `output_config.format` with a `json_schema` rather than text parsing. This guarantees:
- The drafter always returns an `Answer` with citations array (the schema requires it)
- The critic always returns a verdict with `grounded: bool`

No regex parsing. No retry loops on malformed JSON. The schema is the contract.

For Pydantic-defined schemas, the SDK's `client.messages.parse()` method is even cleaner — it auto-generates the schema and validates the response. We chose `messages.create()` + explicit schema for the drafter to demonstrate the lower-level API; in production we'd switch to `parse()`.

## Vector store choice

Chroma in embedded mode. File-based persistence under `data/chroma_db/`. Reasoning:
- Anyone can clone and run the repo with no infra setup
- The `VectorStore` interface (implicit in retrieve.py) lets us swap to pgvector or Pinecone with a thin shim
- For < 100K embeddings, the perf difference vs pgvector is irrelevant

For production, the swap targets are:
- **pgvector** if compliance constrains data residency (one Postgres instance, one place for backups)
- **Pinecone / Weaviate** if scale requires it (>500K embeddings)
- **Voyage embeddings** in place of OpenAI's if you need higher recall on legal/compliance domain

## Corpus profile abstraction

`Corpus` (in `corpora/base.py`) is an ABC with three methods:
- `expected_source_path()` — where the source file lives
- `is_available()` — does the file exist
- `load_chunks()` — produce `CorpusChunk`s

Adding a new framework (ISO 27001, HIPAA, PCI DSS) is one new Corpus subclass + an entry in `get_corpus()`. Everything downstream (ingest, retrieve, agent, eval) is framework-agnostic.

NIST gets a CSV-based loader; AICPA TSC gets a PDF-based loader (when the user supplies a PDF). The interface hides the difference.

## What this is and isn't

It's a reference implementation showing how the pieces fit together. About 1,300 lines of Python including comments. It's not a product. The README and the writeup both lean into that.
