"""Command-line interface for PDF QA system.

Usage:
    python -m src.cli.main "path/to/pdf.pdf"
    python -m src.cli.main "path/to/pdf.pdf" "What is this about?"
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agent import get_agent
from src.config import get_settings


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """PDF QA System - Ask questions about your PDFs using AI.

    Uses GLM-4.5 Flash (free) for text understanding and
    GLM-4.6V Flash for image understanding.
    """
    pass


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--question", "-q", help="Question to ask about the PDF")
@click.option("--interactive", "-i", is_flag=True, help="Interactive Q&A mode")
@click.option("--reindex", is_flag=True, help="Force re-indexing the PDF")
def ask(pdf_path: str, question: Optional[str], interactive: bool, reindex: bool):
    """Ask a question about a PDF.

    Example:
        pdfqa.py ask document.pdf -q "What is the main conclusion?"
    """
    async def run():
        agent = get_agent(pdf_path)

        # Index the PDF (will use cache if available)
        if reindex:
            click.echo(f"\nRe-indexing {Path(pdf_path).name}...")
        else:
            click.echo(f"\nLoading {Path(pdf_path).name}...")

        await agent.index_pdf(force=reindex)

        if reindex:
            click.echo("Indexing complete!")
        else:
            click.echo("Ready!")

        if interactive:
            # Interactive mode
            click.echo("\n" + "="*60)
            click.echo("INTERACTIVE MODE")
            click.echo("Type 'quit' or 'exit' to stop")
            click.echo("="*60 + "\n")

            while True:
                try:
                    q = click.prompt(click.style("Question", fg="cyan"))

                    if q.lower() in ["quit", "exit", "q"]:
                        click.echo("Goodbye!")
                        break

                    result = await agent.ask(q)

                    click.echo("\n" + "="*60)
                    click.echo("ANSWER:")
                    click.echo("="*60)
                    click.echo(result["answer"])
                    click.echo(f"\nSources: Page {result['sources']}")

                except (EOFError, KeyboardInterrupt):
                    click.echo("\nGoodbye!")
                    break

        elif question:
            # Single question mode
            result = await agent.ask(question)

            click.echo("\n" + "="*60)
            click.echo("ANSWER:")
            click.echo("="*60)
            click.echo(result["answer"])
            click.echo(f"\nSources: Page {result['sources']}")
            click.echo(f"Predicted: {result['predicted_pages']}")

        else:
            click.echo("Please provide a question with --question or use --interactive mode")

    asyncio.run(run())


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output JSON file")
@click.option("--reindex", is_flag=True, help="Force re-indexing even if cache exists")
def index(pdf_path: str, output: Optional[str], reindex: bool):
    """Index a PDF for faster Q&A.

    This processes the PDF and generates section summaries.
    Cache is automatically used for subsequent runs unless --reindex is passed.
    """
    async def run():
        agent = get_agent(pdf_path)

        if reindex:
            click.echo(f"\nRe-indexing {Path(pdf_path).name}...")
        else:
            click.echo(f"\nIndexing {Path(pdf_path).name}...")

        summaries = await agent.index_pdf(force=reindex)

        if reindex:
            click.echo(f"Indexed {len(summaries)} sections!")
        else:
            click.echo(f"Loaded {len(summaries)} sections from cache!")

        for s in summaries:
            click.echo(f"\n  Section {s['section_id']} (Pages {s['page_range'][0]}-{s['page_range'][1]}):")
            for point in s.get("summary", [])[:3]:
                click.echo(f"    • {point}")

    asyncio.run(run())


@cli.command()
def config():
    """Show current configuration."""
    settings = get_settings()

    click.echo("\n" + "="*60)
    click.echo("PDF QA SYSTEM CONFIGURATION")
    click.echo("="*60)

    click.echo(f"\nAPI Key: {settings.glm_api_key[:20]}...{settings.glm_api_key[-10:]}")
    click.echo(f"Base URL: {settings.glm_base_url}")
    click.echo(f"Text Model: {settings.glm_model}")
    click.echo(f"Vision Model: {settings.glm_vision_model}")
    click.echo(f"\nProcessing:")
    click.echo(f"   Chunk Size: {settings.chunk_size} pages")
    click.echo(f"   Max Concurrent Calls: {settings.max_concurrent_calls}")
    click.echo(f"   Max Retries: {settings.max_retry_attempts}")
    click.echo(f"   Router Max Sections: {settings.router_max_sections} (0 = all)")

    click.echo(f"\nDirectories:")
    click.echo(f"   Project Root: {settings.project_root}")
    click.echo(f"   PDFs: {settings.pdfs_dir}")
    click.echo(f"   Extracted: {settings.extracted_dir}")

    click.echo("\n" + "="*60 + "\n")


if __name__ == "__main__":
    cli()
