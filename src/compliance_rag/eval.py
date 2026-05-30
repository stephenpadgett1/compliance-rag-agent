"""Eval harness — LLM-as-judge over a 30-case fixture set.

Why LLM-as-judge:
- Compliance answers are open-ended; exact-match graders don't work.
- Faithfulness (claims grounded in citations) and citation accuracy (citations
  actually support the claims) are the two failure modes we care about.
- Opus 4.7 is the judge model — for grading, correctness beats speed/cost.

Caching:
- The judge's instructions are stable across all cases. We put a cache_control
  breakpoint on the system prompt so the judge prefix gets a cache hit on every
  case after the first.

Output:
- Per-run JSON written to `evals/results/<timestamp>.json` containing each case's
  Answer plus the judge's verdict.
- A summary table printed to stdout: mean faithfulness, citation accuracy, etc.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import anthropic
import yaml
from rich.console import Console
from rich.table import Table

from compliance_rag.agent import Agent
from compliance_rag.config import REPO_ROOT, Config
from compliance_rag.retrieve import Retriever
from compliance_rag.schemas import Answer, EvalCase, JudgeVerdict
from compliance_rag.usage import CallStats, UsageStats

console = Console()

JUDGE_INSTRUCTIONS = """You are a strict, fair evaluator of compliance-domain Q&A. Score the \
candidate answer against three criteria. Each score MUST be an integer between 0 and 5 (inclusive).

Scales:
- 5 = excellent / fully supported
- 4 = strong / minor issues only
- 3 = passable / some issues
- 2 = weak / multiple issues
- 1 = poor / largely unsupported
- 0 = no support at all / hallucinated

Criteria:
1. faithfulness — Are the claims in the answer grounded in the cited source excerpts? \
A claim that goes beyond what the citations actually say should lose points.

2. citation_accuracy — Do the cited excerpts actually support the claims they are paired with? \
A wrong-but-real citation (real criterion, doesn't support the claim) should lose points.

3. usefulness — Is the answer useful to the asker? Specific and actionable scores high; \
evasive ("the framework doesn't say") only scores high when the framework genuinely doesn't say. \
Over-hedging should lose points.

Also note: the question may include "expected_criteria" the answer ideally should cite. Use \
these as a hint — perfect coverage of expected criteria is a positive signal, but missing one \
is not automatic failure if the answer is still correct.

Return JSON: {
  "faithfulness": <0-5>,
  "citation_accuracy": <0-5>,
  "usefulness": <0-5>,
  "notes": "2-3 sentence explanation"
}
"""


@dataclass
class CaseResult:
    case: EvalCase
    answer: Answer
    verdict: JudgeVerdict
    agent_usage: UsageStats
    judge_usage: UsageStats

    @property
    def total_usage(self) -> UsageStats:
        return self.agent_usage + self.judge_usage


def load_cases(path: Path | None = None) -> list[EvalCase]:
    """Load eval cases from YAML."""
    if path is None:
        path = REPO_ROOT / "evals" / "cases.yaml"
    raw = yaml.safe_load(path.read_text())
    return [EvalCase.model_validate(c) for c in raw["cases"]]


def judge_one(
    client: anthropic.Anthropic,
    model: str,
    case: EvalCase,
    answer: Answer,
) -> tuple[JudgeVerdict, UsageStats]:
    """Call the judge model on one (case, answer) pair; return verdict + usage."""
    judge_input = (
        f"Question:\n{case.question}\n\n"
        f"Question type: {case.question_type.value}\n"
        f"Expected criteria: {case.expected_criteria}\n"
        f"Grader hint: {case.notes or 'none'}\n\n"
        f"Candidate answer:\n{answer.answer}\n\n"
        f"Candidate citations:\n{json.dumps([c.model_dump() for c in answer.citations], indent=2)}"
    )

    # Note on caching: JUDGE_INSTRUCTIONS is ~400 tokens; Opus 4.7's minimum
    # cacheable prefix is 4096 tokens, so a cache_control marker here would be
    # silently no-op'd by the API. We omit it rather than emit misleading config.
    # The judge is cheap enough per call that this doesn't matter operationally.
    response = client.messages.create(
        model=model,
        max_tokens=2000,
        thinking={"type": "adaptive"},
        system=JUDGE_INSTRUCTIONS,
        messages=[{"role": "user", "content": judge_input}],
        output_config={
            "format": {
                "type": "json_schema",
                # NOTE: Anthropic structured outputs do not support `minimum`/`maximum`
                # on integers. The 0-5 range is enforced by `JudgeVerdict` (Pydantic),
                # not by the JSON schema sent to the API.
                "schema": {
                    "type": "object",
                    "properties": {
                        "faithfulness": {"type": "integer"},
                        "citation_accuracy": {"type": "integer"},
                        "usefulness": {"type": "integer"},
                        "notes": {"type": "string"},
                    },
                    "required": ["faithfulness", "citation_accuracy", "usefulness", "notes"],
                    "additionalProperties": False,
                },
            }
        },
    )
    text = next(b.text for b in response.content if b.type == "text")
    parsed = json.loads(text)
    usage = UsageStats()
    usage.record(CallStats.from_anthropic_message(response, model))
    return JudgeVerdict(case_id=case.id, **parsed), usage


def run_eval(config: Config | None = None, limit: int | None = None) -> list[CaseResult]:
    """Run agent + judge across all cases, return per-case results.

    `limit` truncates to the first N cases — useful for smoke-testing instrumentation
    without paying for a full eval run.
    """
    cfg = config or Config.from_env()
    cases = load_cases()
    if limit is not None:
        cases = cases[:limit]
    console.print(f"[bold]Running eval[/bold] across {len(cases)} cases...")

    # Instantiate Agent + Retriever ONCE — the compiled graph and all clients
    # (Anthropic, OpenAI, Chroma) are reused across every case.
    retriever = Retriever(cfg)
    agent = Agent(config=cfg, retriever=retriever)
    judge_client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)

    results: list[CaseResult] = []
    for case in cases:
        console.print(f"  [{case.id}] {case.question[:60]}...")
        try:
            answer, agent_usage = agent.run(case.question)
            verdict, judge_usage = judge_one(judge_client, cfg.judge_model, case, answer)
            results.append(
                CaseResult(
                    case=case,
                    answer=answer,
                    verdict=verdict,
                    agent_usage=agent_usage,
                    judge_usage=judge_usage,
                )
            )
        except Exception as e:  # noqa: BLE001 — eval suite should continue past individual failures
            console.print(f"    [red]error:[/red] {e}")
            continue

    _write_results(cfg, results)
    _print_summary(results)
    return results


def _write_results(config: Config, results: list[CaseResult]) -> None:
    out_dir = REPO_ROOT / "evals" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{timestamp}.json"

    total_usage = sum((r.total_usage for r in results), start=UsageStats())

    payload = {
        "timestamp": timestamp,
        "corpus": config.corpus,
        "agent_model": config.agent_model,
        "judge_model": config.judge_model,
        "total_usage": total_usage.to_dict(),
        "results": [
            {
                "case": r.case.model_dump(mode="json"),
                "answer": r.answer.model_dump(mode="json"),
                "verdict": r.verdict.model_dump(mode="json"),
                "agent_usage": r.agent_usage.to_dict(),
                "judge_usage": r.judge_usage.to_dict(),
            }
            for r in results
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    console.print(f"\n[dim]Wrote per-case results to {out_path}[/dim]")


def _print_summary(results: list[CaseResult]) -> None:
    if not results:
        console.print("[red]No results to summarize.[/red]")
        return

    table = Table(title="Eval Summary", show_lines=False)
    table.add_column("Type", style="cyan")
    table.add_column("N", justify="right")
    table.add_column("Faithfulness", justify="right")
    table.add_column("Citation Acc.", justify="right")
    table.add_column("Usefulness", justify="right")

    by_type: dict[str, list[CaseResult]] = {}
    for r in results:
        by_type.setdefault(r.case.question_type.value, []).append(r)

    for qtype, group in sorted(by_type.items()):
        n = len(group)
        faith = sum(r.verdict.faithfulness for r in group) / n
        cite = sum(r.verdict.citation_accuracy for r in group) / n
        useful = sum(r.verdict.usefulness for r in group) / n
        table.add_row(qtype, str(n), f"{faith:.2f}", f"{cite:.2f}", f"{useful:.2f}")

    # Overall row
    n = len(results)
    faith = sum(r.verdict.faithfulness for r in results) / n
    cite = sum(r.verdict.citation_accuracy for r in results) / n
    useful = sum(r.verdict.usefulness for r in results) / n
    table.add_row(
        "[bold]OVERALL[/bold]", str(n), f"[bold]{faith:.2f}[/bold]",
        f"[bold]{cite:.2f}[/bold]", f"[bold]{useful:.2f}[/bold]"
    )

    console.print()
    console.print(table)

    # Usage / cost summary
    total = sum((r.total_usage for r in results), start=UsageStats())
    agent_only = sum((r.agent_usage for r in results), start=UsageStats())
    judge_only = sum((r.judge_usage for r in results), start=UsageStats())

    usage_table = Table(title="Token usage and cost", show_lines=False)
    usage_table.add_column("Component", style="cyan")
    usage_table.add_column("Input tokens", justify="right")
    usage_table.add_column("Output tokens", justify="right")
    usage_table.add_column("Cache writes", justify="right")
    usage_table.add_column("Cache reads", justify="right")
    usage_table.add_column("Cost (USD)", justify="right")

    for label, u in [("Agent", agent_only), ("Judge", judge_only), ("[bold]Total[/bold]", total)]:
        usage_table.add_row(
            label,
            f"{u.total_input_tokens:,}",
            f"{u.total_output_tokens:,}",
            f"{u.total_cache_creation_tokens:,}",
            f"{u.total_cache_read_tokens:,}",
            f"${u.total_cost_usd():.4f}",
        )

    console.print()
    console.print(usage_table)
    console.print(
        f"\n[dim]Cache hit rate (cache_read / all_input): "
        f"{total.cache_hit_rate * 100:.1f}%[/dim]"
    )
