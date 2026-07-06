"""
DevBrain CLI — Interactive Q&A in the terminal.

This is the user-facing script for Phase 2. It provides a simple
chat-like interface where you type questions and get cited answers.

Run with:
  uv run python -m scripts.ask

Features:
  - Interactive loop (keep asking questions)
  - Pretty-printed answers with markdown rendering
  - Source citations with clickable URLs
  - Type 'quit' or 'exit' to stop
  - Type 'sources' to toggle showing/hiding source details
"""

import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.rag_chain import RAGChain

console = Console()

# ── Suggested questions ───────────────────────────────────────────
# Shown when the user starts the CLI, to give them ideas
SUGGESTIONS = [
    "How do I add CORS middleware to FastAPI?",
    "How does dependency injection work in FastAPI?",
    "How do I handle file uploads in FastAPI?",
    "What is the difference between Query and Path parameters?",
    "How do I add authentication with OAuth2?",
]


def main():
    """Run the interactive CLI."""
    console.print(Panel(
        "[bold]🧠 DevBrain — FastAPI Developer Assistant[/bold]\n\n"
        "Ask questions about FastAPI. Answers are grounded in official\n"
        "documentation and GitHub issues, with citations.\n\n"
        "[dim]Commands: 'quit' to exit | 'clear' to clear screen[/dim]",
        style="blue",
    ))

    # Show suggested questions
    console.print("\n[bold]💡 Try asking:[/bold]")
    for q in SUGGESTIONS:
        console.print(f"  • {q}")
    console.print()

    # Initialize chain (connects to Pinecone)
    console.print("[dim]Connecting to knowledge base...[/dim]")
    chain = RAGChain(top_k=5, use_mmr=True)
    console.print("[green]Ready![/green]\n")

    show_sources = True

    # ── Interactive loop ──────────────────────────────────────────
    while True:
        try:
            # Get user input
            question = console.input("[bold cyan]You:[/bold cyan] ").strip()

            # Handle commands
            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                console.print("\n[dim]Goodbye! 👋[/dim]")
                break
            if question.lower() == "clear":
                console.clear()
                continue
            if question.lower() == "sources":
                show_sources = not show_sources
                state = "shown" if show_sources else "hidden"
                console.print(f"[dim]Sources are now {state}[/dim]\n")
                continue

            # Ask the question
            console.print()
            console.print("[dim]Thinking...[/dim]")

            result = chain.ask(question)

            # Display the answer
            console.print()
            console.print(Panel(
                Markdown(result["answer"]),
                title="[bold green]DevBrain[/bold green]",
                border_style="green",
                padding=(1, 2),
            ))

            # Display sources
            if show_sources and result["sources"]:
                console.print("\n[bold]📚 Sources:[/bold]")
                for s in result["sources"]:
                    source_type = "📄" if s["source_type"] == "docs" else "🐛"
                    console.print(
                        f"  {source_type} {s['title']} "
                        f"[dim](score: {s['score']})[/dim]"
                    )
                    if s["url"]:
                        console.print(f"     [dim]{s['url']}[/dim]")

            console.print(
                f"\n[dim]Model: {result['model']} | "
                f"Retrieved: {result['num_chunks_retrieved']} chunks[/dim]\n"
            )

        except KeyboardInterrupt:
            console.print("\n\n[dim]Goodbye! 👋[/dim]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")


if __name__ == "__main__":
    main()