"""Command-line entrypoint."""

from __future__ import annotations

import urllib.request

import click
from rich.console import Console

from compliance_rag.config import Config
from compliance_rag.corpora import get_corpus
from compliance_rag.ingest import run_ingest

console = Console()


@click.group()
def main() -> None:
    """Compliance RAG — answer compliance questions against NIST 800-53 or AICPA TSC."""


@main.command()
def status() -> None:
    """Show config and corpus availability. Runs even when API keys are missing."""
    import os

    corpus_name = os.environ.get("CORPUS", "nist-800-53")
    corpus = get_corpus(corpus_name)

    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))

    console.print(f"[bold]Corpus:[/bold] {corpus.display_name}")
    console.print(f"[bold]Source:[/bold] {corpus.expected_source_path()}")
    console.print(f"[bold]Source present:[/bold] {'✓' if corpus.is_available() else '✗'}")
    console.print(f"[bold]Agent model:[/bold] {os.environ.get('AGENT_MODEL', 'claude-sonnet-4-6')}")
    console.print(f"[bold]Judge model:[/bold] {os.environ.get('JUDGE_MODEL', 'claude-opus-4-7')}")
    console.print(f"[bold]ANTHROPIC_API_KEY:[/bold] {'✓ set' if has_anthropic else '[red]✗ missing[/red]'}")
    console.print(f"[bold]OPENAI_API_KEY:[/bold] {'✓ set' if has_openai else '[red]✗ missing[/red]'}")
    if not corpus.is_available():
        console.print(f"\n[yellow]Note:[/yellow] {corpus.distribution_note}")


@main.command(name="download-corpus")
def download_corpus() -> None:
    """Download the default public corpus (NIST SP 800-53). AICPA TSC must be supplied manually."""
    config = Config.from_env()
    corpus = get_corpus(config.corpus)

    if corpus.name != "nist-800-53":
        raise click.ClickException(
            f"Automatic download is only supported for nist-800-53. "
            f"For '{corpus.name}', {corpus.distribution_note}"
        )

    if corpus.is_available():
        console.print(f"[yellow]Already present:[/yellow] {corpus.expected_source_path()}")
        return

    target = corpus.expected_source_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    assert corpus.source_url is not None  # nist always has one

    console.print(f"Downloading {corpus.display_name}...")
    console.print(f"  From: {corpus.source_url}")
    console.print(f"  To:   {target}")

    urllib.request.urlretrieve(corpus.source_url, target)  # noqa: S310 — HTTPS NIST URL
    console.print(f"[green]✓[/green] Downloaded {target.stat().st_size:,} bytes")


@main.command()
def ingest() -> None:
    """Parse the corpus, chunk, embed, and load into Chroma."""
    config = Config.from_env()
    count = run_ingest(config)
    console.print(f"[bold green]Done.[/bold green] {count} chunks indexed.")


@main.command()
@click.argument("question")
def ask(question: str) -> None:
    """Ask the agent a compliance question."""
    from compliance_rag.agent import answer_question

    config = Config.from_env()
    console.print(f"[dim]Q: {question}[/dim]\n")
    answer = answer_question(question, config=config)

    console.print(f"[bold]A:[/bold] {answer.answer}\n")
    console.print(f"[bold]Confidence:[/bold] {answer.confidence}\n")
    console.print("[bold]Citations:[/bold]")
    for cite in answer.citations:
        page = f" (p. {cite.page_number})" if cite.page_number else ""
        console.print(f"  • [{cite.criterion_id}]{page}: {cite.excerpt!r}")


@main.command(name="eval")
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Run only the first N cases (useful for smoke-testing).",
)
def eval_cmd(limit: int | None) -> None:
    """Run the eval suite — 30 cases across 4 question types, judged by Opus 4.7."""
    from compliance_rag.eval import run_eval

    run_eval(Config.from_env(), limit=limit)


if __name__ == "__main__":
    main()
