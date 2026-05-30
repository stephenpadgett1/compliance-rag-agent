"""Usage telemetry — capture token counts, cache hits, and cost per LLM call.

Why this exists:
- The codebase claims `cache_control` saves money on the retrieval context. This
  module makes that claim measurable. Run the eval and look at the cache hit rate.
- "How much did this run cost?" is one of the first questions you get asked about
  any LLM-powered tool. Better to surface it than estimate.

The dataclass is summable (`a + b` returns the merged total), so accumulating
across an agent loop or an eval run is just `sum(stats_list, start=UsageStats())`.

Pricing is cached from the Claude API skill (anchored 2026-04-29). Update if
Anthropic changes per-token pricing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Per-1M-token pricing. Anthropic models from shared/models.md (cache: 2026-04-29).
# OpenAI embedding pricing from the OpenAI docs as of this writing.
PRICING_PER_MILLION: dict[str, tuple[float, float]] = {
    # (input_price, output_price) in USD per 1M tokens
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    # Embeddings: only input is billed; output_price is 0 by convention
    "text-embedding-3-small": (0.02, 0.00),
    "text-embedding-3-large": (0.13, 0.00),
}

# Cache pricing multipliers (relative to base input price)
CACHE_WRITE_MULTIPLIER = 1.25  # 5-minute TTL (default)
CACHE_READ_MULTIPLIER = 0.10


@dataclass
class CallStats:
    """Stats from one LLM call. Created from an Anthropic Message via `from_message`."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @classmethod
    def from_anthropic_message(cls, message: Any, model: str) -> CallStats:
        """Build a CallStats from an Anthropic SDK `Message` object's `.usage`."""
        usage = message.usage
        return cls(
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            # These two are populated when the request includes `cache_control`.
            # The SDK exposes them as attributes that may be None if uncached.
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )

    @classmethod
    def from_openai_embedding(cls, model: str, total_tokens: int) -> CallStats:
        """Build a CallStats for an OpenAI embeddings call."""
        return cls(model=model, input_tokens=total_tokens)

    def cost_usd(self) -> float:
        """Estimate cost in USD for this single call."""
        if self.model not in PRICING_PER_MILLION:
            return 0.0  # unknown model; cost shows as $0 rather than crashing
        input_price, output_price = PRICING_PER_MILLION[self.model]
        cost = self.input_tokens * input_price / 1_000_000
        cost += self.output_tokens * output_price / 1_000_000
        cost += self.cache_creation_input_tokens * input_price * CACHE_WRITE_MULTIPLIER / 1_000_000
        cost += self.cache_read_input_tokens * input_price * CACHE_READ_MULTIPLIER / 1_000_000
        return cost


@dataclass
class UsageStats:
    """Aggregated stats across one or more LLM calls.

    Stored as a list of `CallStats` so we can compute model-specific cost and
    summarize cache behavior accurately. Use `+=` or `+` to merge.
    """

    calls: list[CallStats] = field(default_factory=list)

    def record(self, stats: CallStats) -> None:
        self.calls.append(stats)

    def __add__(self, other: UsageStats) -> UsageStats:
        return UsageStats(calls=self.calls + other.calls)

    def __iadd__(self, other: UsageStats) -> UsageStats:
        self.calls.extend(other.calls)
        return self

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def total_cache_creation_tokens(self) -> int:
        return sum(c.cache_creation_input_tokens for c in self.calls)

    @property
    def total_cache_read_tokens(self) -> int:
        return sum(c.cache_read_input_tokens for c in self.calls)

    @property
    def total_input_billable_tokens(self) -> int:
        """All input tokens, regardless of cache state."""
        return (
            self.total_input_tokens
            + self.total_cache_creation_tokens
            + self.total_cache_read_tokens
        )

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of input tokens served from cache (cache_read / all_input)."""
        total = self.total_input_billable_tokens
        if total == 0:
            return 0.0
        return self.total_cache_read_tokens / total

    def total_cost_usd(self) -> float:
        return sum(c.cost_usd() for c in self.calls)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for inclusion in eval results JSON."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cache_creation_tokens": self.total_cache_creation_tokens,
            "total_cache_read_tokens": self.total_cache_read_tokens,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "total_cost_usd": round(self.total_cost_usd(), 6),
            "calls": [
                {
                    "model": c.model,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "cache_creation_input_tokens": c.cache_creation_input_tokens,
                    "cache_read_input_tokens": c.cache_read_input_tokens,
                }
                for c in self.calls
            ],
        }
