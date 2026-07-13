"""
DevBrain Agent CLI — Interactive Q&A with intelligent routing.

Unlike scripts/ask.py (basic RAG chain), this CLI uses the
LangGraph agent that:
  - Classifies your question type
  - Transforms the query for better retrieval
  - Routes to the right data source
  - Self-corrects when retrieval is poor

Run with:
  uv run python -m scripts.ask_agent
"""

import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.graph import create_agent

console = Console()

SUGGESTIONS = [
    "How do I add CORS middleware to FastAPI?",
    "I'm getting a 422 error with nested Pydantic models",
    "Compare Query parameters vs Path parameters",
    "What is dependency injection in FastAPI?",
    "BackgroundTasks not running after response is sent",
]


def main():
    console.print(Panel(
        "[bold]🤖 DevBrain Agent — Intelligent FastAPI Assistant[/bold]\n\n"
        "This uses an AI agent that classifies your question,\n"
        "routes to the right source, and self-corrects.\n\n"
        "[dim]Commands: 'quit' to exit | 'trace' to toggle agent trace[/dim]",
        style="blue",
    ))

    console.print("\n[bold]💡 Try asking:[/bold]")
    for q in SUGGESTIONS:
        console.print(f"  • {q}")
    console.print()

    console.print("[dim]Initializing agent (building BM25 index)...[/dim]")
    agent = create_agent()
    console.print("[green]Ready![/green]\n")

    show_trace = True

    while True:
        try:
            question = console.input("[bold cyan]You:[/bold cyan] ").strip()

            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                console.print("\n[dim]Goodbye! 👋[/dim]")
                break
            if question.lower() == "trace":
                show_trace = not show_trace
                console.print(f"[dim]Trace {'shown' if show_trace else 'hidden'}[/dim]\n")
                continue

            console.print()

            # Build initial state
            initial_state = {
                "question": question,
                "query_type": "",
                "transformed_queries": [],
                "target_namespaces": [],
                "retrieved_chunks": [],
                "retrieval_grade": "",
                "retry_count": 0,
                "searched_namespaces": [],
                "answer": "",
                "sources": [],
                "trace": [],
            }

            # Run the agent
            result = agent.invoke(initial_state)

            # Display answer
            console.print()
            console.print(Panel(
                Markdown(result["answer"]),
                title="[bold green]DevBrain Agent[/bold green]",
                border_style="green",
                padding=(1, 2),
            ))

            # Display sources
            if result["sources"]:
                console.print("\n[bold]📚 Sources:[/bold]")
                for s in result["sources"]:
                    icon = "📄" if s["source_type"] == "docs" else "🐛"
                    console.print(f"  {icon} {s['title']}")
                    if s["url"]:
                        console.print(f"     [dim]{s['url']}[/dim]")

            # Display trace
            if show_trace and result["trace"]:
                console.print("\n[bold]🔍 Agent Trace:[/bold]")
                for step in result["trace"]:
                    console.print(f"  [dim]→ {step}[/dim]")

            console.print()

        except KeyboardInterrupt:
            console.print("\n\n[dim]Goodbye! 👋[/dim]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")


if __name__ == "__main__":
    main()