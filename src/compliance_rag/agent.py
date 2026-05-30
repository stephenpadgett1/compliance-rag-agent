"""LangGraph agent — planner → retriever → drafter → critic → finalizer.

Design choices grounded in the Anthropic Claude API skill:

1. **LangGraph for state, Anthropic SDK direct for LLM calls.** LangGraph gives
   clean state-machine semantics; calling the SDK directly inside each node lets
   us control `cache_control` placement on the retrieval context.

2. **Prompt caching on the retrieval context.** The retrieved chunks are the
   prime cache target *within* one question's agent loop — drafter and critic
   both see the same chunks, so the second call reads from cache. We put the
   `cache_control: {type: "ephemeral"}` breakpoint on the last system block
   (which caches tools + system together per the prefix-match rule).

3. **Structured output via `output_config.format`.** The drafter is constrained
   to emit the `Answer` schema, so every response has citations. We use
   `client.messages.parse()` for schema-validated decoding.

4. **Adaptive thinking on Sonnet 4.6.** Compliance reasoning is the kind of task
   that benefits from extended thinking — we let the model decide when to use it.

5. **Critic loop, bounded.** The critic checks groundedness. If it fails, we
   loop back to the retriever with an expanded query (max 2 retries to prevent
   runaways).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

import anthropic
from anthropic.types import TextBlockParam
from langgraph.graph import END, StateGraph
from rich.console import Console

from compliance_rag.config import Config
from compliance_rag.corpora.base import CorpusChunk
from compliance_rag.retrieve import Retriever, RetrievalResult
from compliance_rag.schemas import Answer, Citation
from compliance_rag.usage import CallStats, UsageStats

console = Console()

MAX_RETRY_LOOPS = 2
RETRIEVAL_TOP_K = 5


# ---------- System prompts ----------

DRAFTER_INSTRUCTIONS = """You are a compliance expert. You answer questions about IT-governance \
frameworks (NIST SP 800-53, AICPA Trust Services Criteria) based ONLY on the source material \
provided in this conversation.

Rules:
- Cite every load-bearing claim. Each citation must include the criterion ID, a short verbatim \
excerpt from the source, and (if available) the page number.
- Use plain English. Do not use legalese unless quoting source material.
- Be specific: prefer "X must happen within Y days" over "X must happen periodically."
- If the source material does not cover the question, say so explicitly. Do not invent.
- Keep answers to 2-4 paragraphs unless the question genuinely requires more.
"""

CRITIC_INSTRUCTIONS = """You are a strict reviewer. Given a question, an answer, and the source \
material the answerer had access to, check whether every load-bearing claim in the answer is \
supported by a cited excerpt from the source.

Return a JSON object: {
  "grounded": true | false,
  "reasoning": "1-2 sentences explaining your decision",
  "expand_query_with": "string of additional terms to add to retrieval if not grounded, or empty"
}
"""


# ---------- LangGraph state ----------

class AgentState(TypedDict, total=False):
    question: str
    retrieved: list[RetrievalResult]
    draft: Answer
    critic_verdict: dict[str, str | bool]
    iteration: int
    final: Answer
    # `usage` is mutated in place across nodes (one shared UsageStats object).
    usage: UsageStats


# ---------- Graph nodes ----------

@dataclass
class Agent:
    """The compiled LangGraph agent + its dependencies."""

    config: Config
    retriever: Retriever
    anthropic_client: anthropic.Anthropic = field(init=False)

    def __post_init__(self) -> None:
        self.anthropic_client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
        # Compile the graph lazily and cache it — reused across .run() calls.
        self._compiled_graph: Any = None

    def build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("draft", self._draft_node)
        graph.add_node("critic", self._critic_node)
        graph.add_node("finalize", self._finalize_node)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "draft")
        graph.add_edge("draft", "critic")
        graph.add_conditional_edges(
            "critic",
            self._critic_router,
            {"retry": "retrieve", "done": "finalize"},
        )
        graph.add_edge("finalize", END)
        return graph

    def run(self, question: str) -> tuple[Answer, UsageStats]:
        """Run the agent on one question and return the final answer + usage stats.

        Use this in tight loops (e.g. eval) — it reuses the compiled graph and
        all underlying clients (Anthropic, OpenAI, Chroma).
        """
        if self._compiled_graph is None:
            self._compiled_graph = self.build_graph().compile()
        usage = UsageStats()
        result = self._compiled_graph.invoke(
            {"question": question, "iteration": 0, "usage": usage}
        )
        final = result.get("final") or result.get("draft")
        if final is None:
            raise RuntimeError("Agent completed without producing an answer")
        return final, usage

    # --- nodes ---

    def _retrieve_node(self, state: AgentState) -> AgentState:
        question = state["question"]
        prior_verdict = state.get("critic_verdict") or {}
        expand = prior_verdict.get("expand_query_with", "") if isinstance(prior_verdict, dict) else ""
        effective_query = question if not expand else f"{question} {expand}"
        results, embedding_stats = self.retriever.retrieve_with_stats(
            effective_query, top_k=RETRIEVAL_TOP_K
        )
        state["usage"] += embedding_stats
        return {"retrieved": results, "iteration": state.get("iteration", 0) + 1}

    def _draft_node(self, state: AgentState) -> AgentState:
        question = state["question"]
        chunks = [r.chunk for r in state["retrieved"]]
        # System contains ONLY the cached chunks — identical block structure
        # between drafter and critic so the cache key matches. Per-call
        # instructions live in the user message instead.
        system_blocks = self._build_system_blocks(chunks)
        user_content = f"{DRAFTER_INSTRUCTIONS}\n\nQuestion:\n{question}"

        # Schema-constrained output: every response is a valid Answer.
        # - cache_control on the chunks block lets the critic call read the
        #   same prefix from cache.
        # - thinking: adaptive lets Sonnet 4.6 decide when to reason at depth.
        response = self.anthropic_client.messages.create(
            model=self.config.agent_model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=system_blocks,
            messages=[{"role": "user", "content": user_content}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": _answer_json_schema(),
                }
            },
        )
        state["usage"].record(CallStats.from_anthropic_message(response, self.config.agent_model))

        text = _first_text_block(response)
        draft = Answer.model_validate_json(text)
        return {"draft": draft}

    def _critic_node(self, state: AgentState) -> AgentState:
        question = state["question"]
        draft = state["draft"]
        chunks = [r.chunk for r in state["retrieved"]]
        # Identical system structure to drafter — only the chunks block.
        # Per-call critic instructions go in the user message so the cache hits.
        system_blocks = self._build_system_blocks(chunks)

        critic_input = (
            f"{CRITIC_INSTRUCTIONS}\n\n"
            f"Question:\n{question}\n\n"
            f"Answer being reviewed:\n{draft.answer}\n\n"
            f"Citations:\n{json.dumps([c.model_dump() for c in draft.citations], indent=2)}"
        )

        # No `thinking` on the critic — the critic is a structured verdict, not
        # a reasoning task. Aligning thinking with the drafter doesn't help
        # caching anyway (see architecture.md → "Why drafter→critic caching
        # doesn't activate within a single question").
        response = self.anthropic_client.messages.create(
            model=self.config.agent_model,
            max_tokens=2000,
            system=system_blocks,
            messages=[{"role": "user", "content": critic_input}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "grounded": {"type": "boolean"},
                            "reasoning": {"type": "string"},
                            "expand_query_with": {"type": "string"},
                        },
                        "required": ["grounded", "reasoning", "expand_query_with"],
                        "additionalProperties": False,
                    },
                }
            },
        )
        state["usage"].record(CallStats.from_anthropic_message(response, self.config.agent_model))
        verdict = json.loads(_first_text_block(response))
        return {"critic_verdict": verdict}

    def _critic_router(self, state: AgentState) -> Literal["retry", "done"]:
        verdict = state.get("critic_verdict") or {}
        grounded = bool(verdict.get("grounded", True))
        iteration = state.get("iteration", 0)
        if grounded or iteration >= MAX_RETRY_LOOPS + 1:
            return "done"
        return "retry"

    def _finalize_node(self, state: AgentState) -> AgentState:
        draft = state["draft"]
        verdict = state.get("critic_verdict") or {}
        # Downgrade confidence if the critic was lukewarm even after we accepted.
        if isinstance(verdict, dict) and not verdict.get("grounded", True):
            draft = draft.model_copy(update={"confidence": "low"})
        return {"final": draft}

    # --- helpers ---

    @staticmethod
    def _build_system_blocks(chunks: list[CorpusChunk]) -> list[TextBlockParam]:
        """System blocks are just the cached chunks — nothing else.

        Both drafter and critic produce IDENTICAL system block structure (one
        text block containing the source material, marked with `cache_control`).
        Per-call instructions live in the user message instead. This identical
        structure is what makes the cache key match across the two calls.

        Earlier iterations of this code put per-call instructions in a second
        system block; even though the cache_control was on the first block, the
        differing block structure produced different cache keys. The first
        instrumented run showed 8836 cache writes / 0 reads — that was the
        symptom. Fix landed here.
        """
        source_material = "\n\n".join(_format_chunk_for_prompt(c) for c in chunks)
        return [
            {
                "type": "text",
                "text": f"=== SOURCE MATERIAL ===\n\n{source_material}\n\n=== END SOURCE MATERIAL ===",
                "cache_control": {"type": "ephemeral"},
            },
        ]


# ---------- Helpers ----------

def _format_chunk_for_prompt(chunk: CorpusChunk) -> str:
    return f"[{chunk.criterion_id}] {chunk.title}\n{chunk.text}"


def _first_text_block(response: anthropic.types.Message) -> str:
    for block in response.content:
        if block.type == "text":
            return block.text
    raise RuntimeError("Model returned no text content; got stop_reason=" + str(response.stop_reason))


def _answer_json_schema() -> dict:
    """JSON schema for the Answer pydantic model. We hand-write a minimal version
    (rather than reflecting from Pydantic) so the schema stays within the subset
    Anthropic's structured outputs supports — no $ref, no anyOf for enums."""
    return {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "answer": {"type": "string"},
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "criterion_id": {"type": "string"},
                        "excerpt": {"type": "string"},
                        "page_number": {"type": ["integer", "null"]},
                    },
                    "required": ["criterion_id", "excerpt", "page_number"],
                    "additionalProperties": False,
                },
            },
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        },
        "required": ["question", "answer", "citations", "confidence"],
        "additionalProperties": False,
    }


# ---------- Public entrypoint ----------

def answer_question(question: str, config: Config | None = None) -> Answer:
    """Run the full agent loop for one question. Thin wrapper around Agent.run().

    Prefer building an Agent once and calling .run() repeatedly when answering
    multiple questions — this helper exists for ad-hoc CLI use.
    """
    cfg = config or Config.from_env()
    retriever = Retriever(cfg)
    agent = Agent(config=cfg, retriever=retriever)
    answer, _ = agent.run(question)
    return answer
