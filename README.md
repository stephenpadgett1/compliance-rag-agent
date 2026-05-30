# Compliance RAG Agent

A retrieval-augmented compliance Q&A agent. Answers questions like *"What evidence would an auditor want for CC6.1?"* against compliance-framework source documents, with grounded citations back to specific criteria and a 30-case eval harness measuring faithfulness and citation accuracy.

Built as a small public reference for how to wire **LangGraph** + **Claude (Sonnet 4.6 for the agent, Opus 4.7 for LLM-as-judge)** + **prompt caching on the retrieval context** + **a real eval harness**, applied to the compliance-automation use case.

## Why this exists

Most RAG demos skip the parts that matter in production:

- **Citations** are an afterthought, not a first-class output contract
- **Evaluation** is "looks good" rather than measured against a fixture set
- **Prompt caching** is left on the table even though the retrieval context is the *exact* shape that benefits most

This repo tries to do those three things deliberately, and uses the AICPA Trust Services Criteria (or NIST SP 800-53 by default — see below) as a non-trivial, real-world corpus.

## Corpus options

| Corpus | Distribution | Default? |
|--------|--------------|----------|
| **NIST SP 800-53 Rev 5** (Security and Privacy Controls) | Fully public | ✅ Yes — works out of the box |
| **AICPA 2017 Trust Services Criteria** (with 2022 revised Points of Focus) | Requires a free AICPA account; not redistributable | ❌ Optional — user supplies PDF |

The codebase is corpus-agnostic. Adding a new framework (ISO 27001, HIPAA, PCI DSS) is one corpus profile + one ingest run.

## Architecture

```
┌──────────────────┐
│ Source PDF       │  NIST 800-53 (public) or AICPA TSC (user-supplied)
└────────┬─────────┘
         │  ingest.py — semantic chunking on criterion boundaries
         ▼
┌──────────────────┐    ┌────────────────────────────┐
│ Chroma           │    │ LangGraph agent            │
│ (embedded,       │◀───┤  planner → retriever       │
│  file-based)     │    │  → critic → finalizer      │
└──────────────────┘    │  with cache_control on the │
                        │  retrieval context         │
                        └─────────┬──────────────────┘
                                  ▼
                        ┌─────────────────────┐
                        │ Answer + citations  │  schema-validated
                        └─────────────────────┘
                                  │
                                  ▼
                        ┌─────────────────────┐
                        │ eval.py             │  30-case fixture
                        │ (LLM-as-judge,      │  → faithfulness, citation
                        │  Opus 4.7)          │     accuracy, useful-vs-evasive
                        └─────────────────────┘
```

## Quick start

```bash
# 1. Install (requires Python 3.11+)
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# Edit .env to add ANTHROPIC_API_KEY and OPENAI_API_KEY

# 3. Download the default corpus (NIST SP 800-53)
python -m compliance_rag.cli download-corpus

# 4. Build the index
python -m compliance_rag.cli ingest

# 5. Ask a question
python -m compliance_rag.cli ask "What evidence would an auditor want for AC-2?"

# 6. Run the eval suite
python -m compliance_rag.cli eval
```

### Using the AICPA TSC instead

The TSC PDF is freely downloadable but requires a free AICPA account login (the AICPA gates distribution; we cannot redistribute the PDF).

1. Sign up: <https://www.aicpa-cima.com/resources/download/2017-trust-services-criteria-with-revised-points-of-focus-2022>
2. Save the PDF to `data/aicpa-tsc-2017.pdf`
3. Set `CORPUS=aicpa-tsc` in `.env`
4. Re-run `ingest`

## Eval

```bash
python -m compliance_rag.cli eval
```

Runs all 30 fixture cases through the agent, scores each with an Opus 4.7 LLM-as-judge across three axes (faithfulness, citation accuracy, usefulness — each 0–5), and writes per-run results to `evals/results/<timestamp>.json` plus a summary table to stdout.

Cost per run: ~$1 (30 cases × ~$0.03 each on the agent + judge combined).

The fixture set is 30 cases spread across four question types:
- **control_mapping** (8) — *"What controls cover MFA for privileged users?"*
- **criteria_interpretation** (8) — *"What's the difference between AC-3(3) and AC-3(4)?"*
- **operational** (7) — *"How often must access reviews happen per AC-2?"*
- **gap_analysis** (7) — *"Does SSO with SAML satisfy AC-7 entirely?"*

Editable at `evals/cases.yaml`.

**Fresh 30-case run** (Sonnet 4.6 agent, Opus 4.7 judge, NIST SP 800-53, instrumented):

| Question type | N | Faithfulness | Citation accuracy | Usefulness |
|---|---|---|---|---|
| Control mapping | 8 | 4.62 | 5.00 | 4.62 |
| Criteria interpretation | 8 | 4.75 | 4.62 | 4.88 |
| Gap analysis | 7 | 4.57 | 4.71 | 4.71 |
| Operational specificity | 7 | 5.00 | 5.00 | 5.00 |
| **Overall** | **30** | **4.73** | **4.83** | **4.80** |

Total cost for the run: **$1.27** (agent + judge). 91k input tokens + 33k output tokens. Cache hit rate within a single eval run is 0% (see `docs/architecture.md` for why) — but cross-run caching within the 5-minute TTL gives ~50% hit rate on consecutive runs, real cost savings in practice.

See `docs/writeup-draft.md` for what the score patterns tell us about the agent's failure modes.

## What's intentionally simple

This is a reference, not a product:

- Embedded Chroma (no Postgres / Pinecone) — swap behind the `VectorStore` interface
- One agent loop, no multi-agent orchestration
- No web UI; CLI only
- No reranking — vector + keyword hybrid is the retrieval ceiling here

What I'd add for production is called out in `docs/architecture.md` (and the writeup at `docs/writeup-draft.md`).

## Repo layout

```
src/compliance_rag/
  ├── corpora/       # corpus profiles (parser, ID format, citation shape)
  ├── ingest.py      # PDF → chunks → embeddings → Chroma
  ├── retrieve.py    # hybrid retrieval
  ├── agent.py       # LangGraph agent
  ├── eval.py        # LLM-as-judge eval harness
  ├── schemas.py     # Pydantic models (Answer, Citation, EvalCase, ...)
  ├── config.py      # env + model IDs
  └── cli.py         # entrypoint
evals/
  ├── cases.yaml     # 30 graded questions
  └── results/       # per-run eval outputs
```

## License

MIT. See `LICENSE`.
