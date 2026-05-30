# What I learned building a SOC 2 RAG agent

*Draft — distribution channel TBD. Author: Stephen Padgett.*

---

I spent a weekend building a small RAG agent that answers compliance questions about NIST SP 800-53 and the AICPA Trust Services Criteria — the kind of system compliance-automation vendors sell to customers. The build is on [GitHub](https://github.com/stephenpadgett1/compliance-rag-agent).

The agent works. The interesting part is what I had to throw away from the standard RAG-demo playbook to get there.

## Most compliance RAG demos optimize for the wrong thing

The default RAG-demo move is to chunk a PDF by token count, embed everything, retrieve top-K, and have the model write a helpful answer. The implicit goal is **be useful** — minimize "I don't know" answers, prefer specific guidance over hedging.

For compliance, that goal is wrong. The right goal is closer to: **be calibrated**. A SOC 2 customer asking "how often must we review access?" deserves the honest answer — *AC-2 defers cadence to organization-defined parameters; pick a frequency and document it* — not a fabricated "quarterly is industry standard." The framework genuinely doesn't say. An agent that pretends otherwise is worse than one that defers.

This is non-obvious because it inverts a tuning instinct most AI engineers have. We've all been told "don't be evasive." For compliance, evasive is sometimes precisely the right answer.

## The corollary is that the cite *is* the citation

When the answer is "the framework defers this," the credibility move is to cite the exact criterion that defers it. Not the page number where the deferral appears — *the criterion ID*. `AC-2(j)` lands differently than "page 42" because the auditor can look up `AC-2(j)` in their own tools, in their own copy, in their own format.

This pushed me toward an architectural choice that ended up being the most consequential one in the build: **use the canonical machine-readable source whenever it exists.** NIST publishes 800-53 as a structured CSV (1190 rows, one per control or enhancement). Parsing it gives me 1009 clean chunks with stable IDs and a known parent-child relationship. Parsing the PDF would have given me approximately the same chunks plus brittle page-boundary heuristics. The PDF path is for frameworks like the AICPA TSC that don't publish anything else.

## Citation accuracy matters more than recall

Compliance work involves audit. Auditors spot-check claims. A 90%-accurate-citations, 70%-recall agent is more useful than a 95%-recall, 60%-accurate-citations agent — because the failure mode of the second one (a confident citation that doesn't actually say what the agent claims) erodes trust faster than the failure mode of the first (missing a relevant control).

I built the eval harness around this asymmetry. The 30-case fixture set grades on three axes: faithfulness (are the claims grounded in the citations?), citation accuracy (do the citations actually support the claims?), and usefulness (specific vs evasive). The judge model (Opus 4.7, separate context window from the agent) scores each on 0–5. I weight citation accuracy slightly higher than usefulness when reading the results — and the agent's prompt is tuned to match.

Results on the first clean run, against NIST SP 800-53:

| Question type | N | Faithfulness | Citation accuracy | Usefulness |
|---|---|---|---|---|
| Control mapping | 8 | 4.62 | **5.00** | 4.62 |
| Criteria interpretation | 8 | 4.75 | 4.88 | 5.00 |
| Gap analysis | 7 | 4.71 | 4.86 | 4.71 |
| Operational specificity | 7 | 4.86 | 4.86 | 5.00 |
| **Overall** | **30** | **4.73** | **4.90** | **4.83** |

82% of all individual scores were 5/5; the rest were 4/5; zero scored 3 or below. The most interesting result is the lowest-scoring cases. Every one was a control-mapping question where the agent cited everything it cited correctly but missed one or two controls it should also have surfaced. That's exactly the failure mode the writeup is arguing for: high precision, lower recall, no hallucinated citations. The agent's behavior matches the thesis.

The three perfect-5/5/5 cases were all gap-analysis questions — "is X enough to satisfy Y?" The agent did its best work being honest about what concrete configurations *don't* cover. Which, again, is the point.

## What hybrid retrieval gets you

Pure vector retrieval works fine for questions like "what does the framework say about removable media?" — semantic similarity will surface MP-2, MP-5, MP-7 because the chunks describe removable-media handling.

It falls over for questions like "what's the difference between AC-3(3) and AC-3(4)?" An embedding model doesn't always know that `AC-3(3)` is a specific *string* the user typed and the user wants *that specific string* back. So I added a regex-based ID lookup that runs alongside vector search. Hybrid merge prioritizes ID-direct hits over vector hits, since an explicit ID is a stronger intent signal than a fuzzy match.

This took maybe 40 lines of code and meaningfully improved the eval scores on operational and gap-analysis questions.

## The cache-key bug that the instrumentation caught (and the deeper one it didn't fix)

I claimed in the architecture doc that the retrieval context was prompt-cached across the drafter and critic calls — the obvious win. So I instrumented every API call to read back `usage.cache_creation_input_tokens` and `usage.cache_read_input_tokens`.

The first instrumented run showed **8,836 cache writes and 0 cache reads**.

The first diagnosis took half an hour. Both calls placed the `cache_control` marker on a system block containing the same chunks. The cached *bytes* were identical. But the drafter and critic each had a second system block with their own per-call instructions, and the cache key includes the system block list structure, not just the prefix bytes up to the marker. Different second blocks → different cache keys → no shared cache.

Fix: make the system blocks structurally identical. Each call now sends exactly one system block (the chunks). The per-call instructions moved into the user message, which sits after the cache marker and doesn't affect the key.

That fixed *cross-run* caching — when the same eval cases run back-to-back within the 5-minute TTL, the second run reads ~50% of its input tokens from cache. Real cost reduction.

But the *drafter→critic within a single question* path still showed 0 reads. The two calls now had identical system blocks, but each was writing a different number of cache tokens — 1,926 vs 1,762, a **consistent 164-token delta across every case**. Same chunks, same system block list, but the cache keys still didn't match.

The 164-token delta is consistent with the size difference between the drafter's structured-output schema (returns an `Answer` with citations) and the critic's (returns a verdict). The most likely explanation: `output_config.format.schema` participates in the cache key, even though Anthropic's documented invalidation hierarchy doesn't list it. This isn't fixable without giving up structured outputs on one of the two calls — a contortion worse than the cost savings.

The deeper lesson: I'd have happily shipped this with no telemetry, claiming a cost optimization that didn't exist. Anthropic's API silently accepts a `cache_control` marker and silently doesn't cache when the key changes for reasons not in the docs. "Did you measure it?" is the right question to ask any prompt-caching claim — and "did you measure it across the workload that actually matters?" is the right follow-up.

## What's intentionally simple

This isn't a product. It's a reference. Specifically:

- **Embedded Chroma**, not pgvector or Pinecone. Anyone clones the repo and runs it. Swapping the vector store is one interface implementation.
- **One agent loop**, not multi-agent orchestration. LangGraph for state, the Anthropic SDK called directly inside nodes so I can put `cache_control` on the retrieval context — the second LLM call (the critic) reads the cached chunks, not pays for them again.
- **CLI only**, no web UI. The point is the pipeline shape.

The whole thing is about 1,300 lines of Python including comments.

## What I'd add for production

A real compliance-AI product needs things I deliberately left out:

- **Auditable retrieval trails** — every answer logged with the exact chunks the agent saw, for after-the-fact spot-checks.
- **Per-customer evidence corpora** — a compliance vendor's value isn't the framework, it's mapping the *customer's* policies, screenshots, and tickets to framework criteria. The retrieval shape changes accordingly.
- **Reranking** — for ambiguous queries, a cross-encoder reranker after vector retrieval catches cases where the embedding model gets the topic right but the specific control wrong.
- **Continuous eval** — the 30-case fixture is a starting line. Production needs a growing eval set that captures real customer questions over time.

But the core insight stands regardless of how much you bolt on: **for compliance, calibration beats utility.** Build the system that's right when it says it doesn't know.

---

*Stephen Padgett ran engineering at ISACA, the global standards body for IT governance, audit, and risk certifications (CISA, CISM, CRISC, COBIT). He currently consults on production AI through [Centaur Services](TODO-add-link).*
