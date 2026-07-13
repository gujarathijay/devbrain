"""
Query Transformation — make questions better before searching.

The user's raw question is often not the best search query.
These transformations improve retrieval by rephrasing the question
in ways that match how information is stored in the knowledge base.

Three strategies:

1. HyDE (Hypothetical Document Embedding)
   Ask the LLM to write a hypothetical answer, then search using THAT
   instead of the question. The hypothetical answer uses the same
   vocabulary as the actual docs → better embedding match.

2. Query Decomposition
   Break complex/comparative questions into simpler sub-queries.
   "Compare A vs B" → ["What is A?", "What is B?"]
   Search each separately, combine results.

3. Step-back Prompting
   Make the question more general to retrieve broader context.
   "Why does OAuth2PasswordBearer raise 401?" →
   "How does OAuth2 authentication work in FastAPI?"
"""

import openai
from rich.console import Console

from src.config import OPENAI_API_KEY

console = Console()

client = openai.OpenAI(api_key=OPENAI_API_KEY)
MODEL = "gpt-4o-mini"


def generate_hyde(question: str) -> str:
    """
    HyDE: Generate a hypothetical answer to use as the search query.

    Why this works:
    The user asks: "How do I add CORS?"
    This is a short question → its embedding is broad.

    HyDE generates: "To add CORS in FastAPI, import CORSMiddleware
    from fastapi.middleware.cors. Call app.add_middleware() with
    allow_origins, allow_methods, and allow_headers..."

    This hypothetical answer:
    - Uses the same terms as the actual docs ("CORSMiddleware", "allow_origins")
    - Has similar structure to the actual docs
    - Produces an embedding that's closer to the real docs in vector space

    The hypothetical doesn't need to be correct — it just needs to be
    in the right "neighborhood" of the vector space.
    """
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a FastAPI documentation writer. Given a question, "
                    "write a short paragraph (3-5 sentences) that would answer it, "
                    "as if it were a section in the official FastAPI documentation. "
                    "Use specific FastAPI terminology, class names, and code references. "
                    "Do NOT say 'I don't know' — write a plausible documentation excerpt."
                ),
            },
            {"role": "user", "content": question},
        ],
    )

    hyde_doc = response.choices[0].message.content
    return hyde_doc


def decompose_query(question: str) -> list[str]:
    """
    Break a complex question into simpler sub-queries.

    When to use:
    - Comparison questions: "Compare X vs Y"
    - Multi-part questions: "How does X work and what are its limitations?"
    - Questions spanning multiple topics

    Example:
      Input:  "Compare Query parameters vs Path parameters in FastAPI"
      Output: [
        "What are Query parameters in FastAPI and how do you use them?",
        "What are Path parameters in FastAPI and how do you use them?"
      ]

    We search each sub-query separately and merge results.
    This gives us focused, relevant chunks for each aspect.
    """
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a search query optimizer. Given a complex question, "
                    "break it into 2-3 simpler, self-contained sub-questions that "
                    "together would answer the original question.\n\n"
                    "Rules:\n"
                    "- Each sub-question should be searchable on its own\n"
                    "- Each should focus on one specific aspect\n"
                    "- Return ONLY the sub-questions, one per line\n"
                    "- No numbering, no bullets, no extra text"
                ),
            },
            {"role": "user", "content": question},
        ],
    )

    sub_queries = [
        q.strip()
        for q in response.choices[0].message.content.strip().split("\n")
        if q.strip()
    ]

    return sub_queries


def step_back_query(question: str) -> str:
    """
    Generate a broader version of the question.

    When to use:
    - Very specific questions that might not have direct doc matches
    - Error-specific questions where understanding the broader system helps

    Example:
      Input:  "Why does my OAuth2PasswordBearer return 401 when token is valid?"
      Output: "How does OAuth2 token authentication work in FastAPI?"

    The broader question retrieves the full authentication docs,
    which contain the specific error handling information the user needs.
    """
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a search query optimizer. Given a specific question, "
                    "generate a broader, more general version that would retrieve "
                    "the background documentation needed to answer the original.\n\n"
                    "The step-back question should be about the general concept "
                    "or system, not the specific issue.\n\n"
                    "Return ONLY the broader question, nothing else."
                ),
            },
            {"role": "user", "content": question},
        ],
    )

    return response.choices[0].message.content.strip()


# ── Self-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from rich.panel import Panel

    console.print("\n[bold]🔄 Testing Query Transformations[/bold]\n")

    # Test HyDE
    q1 = "How do I add CORS middleware?"
    console.print(f"[bold]Original:[/bold] {q1}")
    hyde = generate_hyde(q1)
    console.print(Panel(hyde, title="HyDE (Hypothetical Doc)", border_style="green"))

    # Test Decomposition
    q2 = "Compare Query parameters vs Path parameters in FastAPI"
    console.print(f"\n[bold]Original:[/bold] {q2}")
    subs = decompose_query(q2)
    for i, sq in enumerate(subs, 1):
        console.print(f"  Sub-query {i}: {sq}")

    # Test Step-back
    q3 = "Why does OAuth2PasswordBearer raise 401 when my token is valid?"
    console.print(f"\n[bold]Original:[/bold] {q3}")
    stepped = step_back_query(q3)
    console.print(Panel(stepped, title="Step-back Query", border_style="yellow"))