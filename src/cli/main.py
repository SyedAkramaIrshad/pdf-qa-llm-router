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
def ask(pdf_path: str, question: Optional[str], interactive: bool):
    """Ask a question about a PDF.

    Example:
        pdf-ask document.pdf "What is the main conclusion?"
    """
    async def run():
        agent = get_agent(pdf_path)

        # Index the PDF
        click.echo(click.style(f"\n📚 Indexing {Path(pdf_path).name}...", fg="blue"))
        with click.progressbar(length=100, label="Progress") as bar:
            await agent.index_pdf()
            bar.update(100)

        click.echo(click.style("✅ Indexing complete!", fg="green"))

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
                    click.echo(click.style("ANSWER:", fg="green", bold=True))
                    click.echo("="*60)
                    click.echo(result["answer"])
                    click.echo(f"\n📖 Sources: Page {result['sources']}")

                except (EOFError, KeyboardInterrupt):
                    click.echo("\nGoodbye!")
                    break

        elif question:
            # Single question mode
            result = await agent.ask(question)

            click.echo("\n" + "="*60)
            click.echo(click.style("ANSWER:", fg="green", bold=True))
            click.echo("="*60)
            click.echo(result["answer"])
            click.echo(f"\n📖 Sources: Page {result['sources']}")
            click.echo(f"🔍 Predicted: {result['predicted_pages']}")

        else:
            click.echo("Please provide a question with --question or use --interactive mode")

    asyncio.run(run())


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output JSON file")
def index(pdf_path: str, output: Optional[str]):
    """Index a PDF for faster Q&A.

    This processes the PDF and generates section summaries.
    Use this before asking multiple questions.
    """
    async def run():
        agent = get_agent(pdf_path)

        click.echo(click.style(f"\n📚 Indexing {Path(pdf_path).name}...", fg="blue"))

        summaries = await agent.index_pdf()

        click.echo(click.style(f"✅ Indexed {len(summaries)} sections!", fg="green"))

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

    click.echo(f"\n🔑 API Key: {settings.glm_api_key[:20]}...{settings.glm_api_key[-10:]}")
    click.echo(f"🌐 Base URL: {settings.glm_base_url}")
    click.echo(f"🤖 Text Model: {settings.glm_model}")
    click.echo(f"👁️  Vision Model: {settings.glm_vision_model}")
    click.echo(f"\n📊 Processing:")
    click.echo(f"   Chunk Size: {settings.chunk_size} pages")
    click.echo(f"   Max Concurrent Calls: {settings.max_concurrent_calls}")
    click.echo(f"   Max Retries: {settings.max_retry_attempts}")

    click.echo(f"\n📁 Directories:")
    click.echo(f"   Project Root: {settings.project_root}")
    click.echo(f"   PDFs: {settings.pdfs_dir}")
    click.echo(f"   Extracted: {settings.extracted_dir}")

    click.echo("\n" + "="*60 + "\n")


if __name__ == "__main__":
    cli()
